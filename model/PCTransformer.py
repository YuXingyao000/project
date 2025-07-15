import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F

from timm.models.layers import DropPath,trunc_normal_

from DGCNN import DGCNN_Grouper, kNNQuery
import numpy as np

from utils import knn_index, square_distance, fps_downsample
from DGCNN import kNNQuery

class FeedForward(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, dropout_rate=0.0):
        super().__init__()
        hidden_channels = hidden_channels if hidden_channels is not None else in_channels * 2
        
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, in_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
 
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for point cloud processing.
    
    This module implements scaled dot-product attention with multiple heads,
    taking separate Q, K, V inputs for flexibility in self-attention and cross-attention.
    """
    def __init__(self, d_model, num_heads=8, attn_drop=0., proj_drop=0.):
        """
        Initialize the attention module.
        
        Args:
            d_model (int): Input and output feature dimension
            num_heads (int): Number of attention heads
            attn_drop (float): Dropout rate for attention weights
            proj_drop (float): Dropout rate for output projection
        """
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.scale = self.d_k ** -0.5

        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(attn_drop)
        self.output_dropout = nn.Dropout(proj_drop)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """
        Compute attention weights and apply them to values.
        
        Args:
            query (torch.Tensor): Query tensor of shape [batch, heads, seq_len, head_dim]
            key (torch.Tensor): Key tensor of shape [batch, heads, seq_len, head_dim]
            value (torch.Tensor): Value tensor of shape [batch, heads, seq_len, head_dim]
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Attention output of shape [batch, heads, seq_len, head_dim]
        """
        # Compute attention scores: Q * K^T
        attention_scores = torch.einsum('b h i d, b h j d -> b h i j', query, key)
        
        # Scale the attention scores
        attention_scores = attention_scores * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, value)
        
        return attention_output

    def forward(self, q, k, v, mask=None):
        """
        Forward pass of the attention mechanism.
        
        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, seq_len, d_model]
            k (torch.Tensor): Key tensor of shape [batch_size, seq_len, d_model]
            v (torch.Tensor): Value tensor of shape [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = q.shape
        
        # Step 1: Reshape Q, K, V for multi-head attention
        # Rearrange: [batch, seq, d_model] -> [batch, heads, seq, d_k]
        q = rearrange(q, 'batch seq (heads d_k) -> batch heads seq d_k', 
                     heads=self.num_heads, d_k=self.d_k)
        k = rearrange(k, 'batch seq (heads d_k) -> batch heads seq d_k', 
                     heads=self.num_heads, d_k=self.d_k)
        v = rearrange(v, 'batch seq (heads d_k) -> batch heads seq d_k', 
                     heads=self.num_heads, d_k=self.d_k)
        
        # Step 2: Compute attention
        attention_output = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Step 3: Reshape back to original format
        # Rearrange: [batch, heads, seq, d_k] -> [batch, seq, d_model]
        output = rearrange(attention_output, 'batch heads seq d_k -> batch seq (heads d_k)')
        
        # Step 4: Apply output projection and dropout
        output = self.output_projection(output)
        output = self.output_dropout(output)
        
        return output

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model=384, num_heads=6):
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads
            )
        self.feed_forward = FeedForward(d_model, hidden_channels=d_model * 2)
        self.feed_forward_norm = nn.LayerNorm(d_model)
    
    def forward(self, points):
        """
        Forward pass for vanilla transformer block.
        
        Args:
            points (torch.Tensor): Input points with shape [batch, 3+feature_dim, num_points]
            
        Returns:
            torch.Tensor: Output points with shape [batch, 3+feature_dim, num_points]
        """
        # Extract coordinates and features properly
        coords = points[:, :3, :]  # [batch, 3, num_points]
        features = points[:, 3:, :]  # [batch, feature_dim, num_points]
        
        # Normalize features
        norm_features = self.input_norm(features.transpose(1, 2))  # [batch, num_points, feature_dim]
        
        # Self-attention
        qkv = self.qkv_proj(norm_features)  # [batch, num_points, 3*feature_dim]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch, num_points, feature_dim]
        attn_features = self.multi_head_attention(q, k, v)  # [batch, num_points, feature_dim]
        
        # Residual connection
        output_features = features.transpose(1, 2) + attn_features  # [batch, num_points, feature_dim]
        # Feed-forward network
        output_features = self.feed_forward( self.feed_forward_norm(output_features)) + output_features
        
        return torch.cat([coords, output_features.transpose(1, 2)], dim=1)  # [batch, 3+feature_dim, num_points]

class GeometryAwareSelfAttentionBlock(nn.Module):
    def __init__(self, d_model=384, num_heads=6):
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads
            )
        self.feed_forward = FeedForward(d_model, hidden_channels=d_model * 2)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.kNNQuery = kNNQuery(k_nearest_neighbors=8)
        self.kNN_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.merge_proj = nn.Linear(d_model * 2, d_model)
        
    def forward(self, points):
        """
        Forward pass for geometry-aware transformer block.
        
        Args:
            points (torch.Tensor): Input points with shape [batch, 3+feature_dim, num_points]
            
        Returns:
            torch.Tensor: Output points with shape [batch, 3+feature_dim, num_points]
        """
        # Extract coordinates and features properly
        coords = points[:, :3, :]  # [batch, 3, num_points]
        features = points[:, 3:, :]  # [batch, feature_dim, num_points]
        
        # Normalize features
        norm_features = self.input_norm(features.transpose(1, 2))  # [batch, num_points, feature_dim]
        
        # Self-attention
        qkv = self.qkv_proj(norm_features)  # [batch, num_points, 3*feature_dim]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch, num_points, feature_dim]
        attn_features = self.multi_head_attention(q, k, v)  # [batch, num_points, feature_dim]
        
        # Geometry-aware features using kNN
        # Prepare input for kNN: concatenate coords and normalized features
        normed_points = torch.cat([coords, norm_features.transpose(1, 2)], dim=1)  # [batch, 3+feature_dim, num_points]
        
        # Get geometry features
        geom_features = self.kNNQuery(normed_points, normed_points)  # [batch, 2*feature_dim, num_points, k]
        geom_features = rearrange(geom_features, 'batch double_feature_dim num_points k -> batch k num_points double_feature_dim')
        geom_features = self.kNN_proj(geom_features)  # [batch, k, num_points, feature_dim]
        geom_features = geom_features.max(dim=1, keepdim=False)[0]  # [batch, num_points, feature_dim]
        
        # Merge attention and geometry features
        merged_features = torch.cat([attn_features, geom_features], dim=-1)  # [batch, num_points, 2*feature_dim]
        merged_features = self.merge_proj(merged_features)  # [batch, num_points, feature_dim]
        
        # Residual connection
        output_features = merged_features + attn_features  # [batch, num_points, feature_dim]
        
        # Feed-forward network
        output_features = self.feed_forward_norm(output_features)
        output_features = self.feed_forward(output_features) + output_features
        
        # Prepare output: concatenate coords and features
        output = torch.cat([coords, output_features.transpose(1, 2)], dim=1)  # [batch, 3+feature_dim, num_points]
        
        return output

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model=384, num_heads=6, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.query_norm = nn.LayerNorm(d_model)
        self.key_norm = nn.LayerNorm(d_model)
        
        self.feed_forward = FeedForward(d_model, hidden_channels=d_model * 2)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        
        self.q_map = nn.Linear(d_model, d_model, bias=False)
        self.k_map = nn.Linear(d_model, d_model, bias=False)
        self.v_map = nn.Linear(d_model, d_model, bias=False)
        
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads
            )
        
    def forward(self, query_points, key_points):
        query_coords = query_points[:, :3, :]
        key_coords = key_points[:, :3, :]
        query_features = query_points[:, 3:, :]
        key_features = key_points[:, 3:, :]
        
        query_features = self.query_norm(query_features.transpose(1,2))
        key_features = self.key_norm(key_features.transpose(1,2))
        
        q = self.q_map(query_features)
        k = self.k_map(key_features)
        v = self.v_map(key_features)
        attn_features = self.multi_head_attention(q, k, v)
         
        query_features = query_features + attn_features
        query_features = query_features + self.feed_forward(self.feed_forward_norm(query_features))
        
        return torch.cat([query_coords, query_features.transpose(1,2)], dim=1)

class GeometryAwareCrossAttentionBlock(nn.Module):
    def __init__(self, d_model=384, num_heads=6):
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads
            )
        self.feed_forward = FeedForward(d_model, hidden_channels=d_model * 2)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.self_kNNQuery = kNNQuery(k_nearest_neighbors=8)
        self.cross_kNNQuery = kNNQuery(k_nearest_neighbors=8)
        self.kNN_proj1 = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.kNN_proj2 = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.self_merge_proj = nn.Linear(d_model * 2, d_model)
        self.cross_merge_proj = nn.Linear(d_model * 2, d_model)
        self.cross_norm_q = nn.LayerNorm(d_model)
        self.cross_norm_k = nn.LayerNorm(d_model)
        self.cross_q_map = nn.Linear(d_model, d_model, bias=False)
        self.cross_k_map = nn.Linear(d_model, d_model, bias=False)
        self.cross_v_map = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, query_points, key_points):
        """
        Forward pass for geometry-aware transformer block.
        
        Args:
            points (torch.Tensor): Input points with shape [batch, 3+feature_dim, num_points]
            
        Returns:
            torch.Tensor: Output points with shape [batch, 3+feature_dim, num_points]
        """
        #################################
        # Geometry-aware Self-attention #
        #################################
        # Extract coordinates and features properly
        query_coords = query_points[:, :3, :]  # [batch, 3, num_points]
        key_coords = key_points[:, :3, :]  # [batch, 3, num_points]
        query_features = query_points[:, 3:, :]  # [batch, feature_dim, num_points]
        key_features = key_points[:, 3:, :]  # [batch, feature_dim, num_points]
        
        # Normalize features
        norm_features = self.input_norm(query_features.transpose(1, 2))  # [batch, num_points, feature_dim]
        
        # Self-attention
        qkv = self.qkv_proj(norm_features)  # [batch, num_points, 3*feature_dim]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch, num_points, feature_dim]
        attn_features = self.multi_head_attention(q, k, v)  # [batch, num_points, feature_dim]
        
        # Geometry-aware features using kNN
        # Prepare input for kNN: concatenate coords and normalized features
        normed_points = torch.cat([query_coords, norm_features.transpose(1, 2)], dim=1)  # [batch, 3+feature_dim, num_points]
        
        # Get geometry features
        geom_features = self.self_kNNQuery(normed_points, normed_points)  # [batch, 2*feature_dim, num_points, k]
        geom_features = rearrange(geom_features, 'batch double_feature_dim num_points k -> batch k num_points double_feature_dim')
        geom_features = self.kNN_proj1(geom_features)  # [batch, k, num_points, feature_dim]
        geom_features = geom_features.max(dim=1, keepdim=False)[0]  # [batch, num_points, feature_dim]
        
        # Merge attention and geometry features
        attn_features = torch.cat([attn_features, geom_features], dim=-1)  # [batch, num_points, 2*feature_dim]
        attn_features = self.self_merge_proj(attn_features)  # [batch, num_points, feature_dim]
        
        # Residual connection
        query_features = attn_features + query_features.transpose(1,2)  # [batch, num_points, feature_dim]
        
        #################################
        # Geometry-aware Cross-attention #
        #################################
        
        normed_query_features = self.cross_norm_q(query_features)
        normed_key_features = self.cross_norm_k(key_features.transpose(1,2))
        
        cross_q = self.cross_q_map(normed_query_features)
        cross_k = self.cross_k_map(normed_key_features)
        cross_v = self.cross_v_map(normed_key_features)
        cross_features = self.multi_head_attention(cross_q, cross_k, cross_v)
        
        corss_query_points = torch.cat([query_coords, normed_query_features.transpose(1,2)], dim=1)
        cross_key_points = torch.cat([key_coords, normed_key_features.transpose(1,2)], dim=1)
        cross_geom_features = self.cross_kNNQuery(corss_query_points, cross_key_points)
        cross_geom_features = rearrange(cross_geom_features, 'batch double_feature_dim num_points k -> batch k num_points double_feature_dim')
        cross_geom_features = self.kNN_proj2(cross_geom_features)  # [batch, k, num_points, feature_dim]
        cross_geom_features = cross_geom_features.max(dim=1, keepdim=False)[0]  # [batch, num_points, feature_dim]
        
        cross_features = torch.cat([cross_features, cross_geom_features], dim=-1)  # [batch, num_points, 2*feature_dim]
        cross_features = self.cross_merge_proj(cross_features)  # [batch, num_points, feature_dim]
        
        query_features = query_features + cross_features
        query_features = query_features + self.feed_forward(self.feed_forward_norm(query_features))
        
        return torch.cat([query_coords, query_features.transpose(1,2)], dim=1)
        
class GeometryAwareTransformerDecoder(nn.Module):
    def __init__(self, d_model=384, geom_block_num=1, cross_block_num=5, num_heads=6):
        super().__init__()
        self.blocks = nn.ModuleList([
            GeometryAwareCrossAttentionBlock(d_model=d_model, num_heads=num_heads) for i in range(geom_block_num)
        ])
        self.blocks.extend([
            CrossAttentionBlock(d_model=d_model, num_heads=num_heads) for i in range(cross_block_num)
        ])
    def forward(self, query_points, key_points):
        for block in self.blocks:
            query_points = block(query_points, key_points)
        return query_points

class GeometryAwareTransformerEncoder(nn.Module):
    """
    PCTransformer Encoder with geometry-aware attention blocks.
    """
    def __init__(self, d_model=384, geom_block_num=1, vanilla_block_num=5, num_heads=6):
        super().__init__()
        assert geom_block_num + vanilla_block_num > 0, "Block number must be greater than 0"
        self.blocks = nn.ModuleList([
            GeometryAwareSelfAttentionBlock(d_model=d_model, num_heads=num_heads) for i in range(geom_block_num)
        ])
        self.blocks.extend([
            SelfAttentionBlock(d_model=d_model, num_heads=num_heads) for i in range(vanilla_block_num)
        ])
       
    
    def forward(self, points):
        for block in self.blocks:
            points = block(points)
        return points


class PCTransformer(nn.Module):
    """ Vision Transformer with support for point cloud completion
    """
    def __init__(self, in_chans=3, embed_dim=384, depth=[[1, 5], [1, 5]], num_heads=6, num_query = 224):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim
        
        self.grouper = DGCNN_Grouper()  # B 3 N to B C(3) N(128) and B C(128) N(128)

        self.pos_embed = nn.Sequential(
            nn.Conv1d(in_chans, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, embed_dim, 1)
        )

        self.input_proj = nn.Sequential(
            nn.Conv1d(128, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(embed_dim, embed_dim, 1)
        )

        self.encoder = GeometryAwareTransformerEncoder(d_model=embed_dim, geom_block_num=depth[0][0], vanilla_block_num=depth[0][1], num_heads=num_heads)

        self.increase_dim = nn.Sequential(
            nn.Conv1d(embed_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        self.num_query = num_query
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * num_query)
        )
        self.query_conv = nn.Sequential(
            nn.Conv1d(1024 + 3, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, embed_dim, 1)
        )

        self.decoder = GeometryAwareTransformerDecoder(d_model=embed_dim, geom_block_num=depth[1][0], cross_block_num=depth[1][1], num_heads=num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def pos_encoding_sin_wave(self, coor):
        # ref to https://arxiv.org/pdf/2003.08934v2.pdf
        D = 64 #
        # normal the coor into [-1, 1], batch wise
        normal_coor = 2 * ((coor - coor.min()) / (coor.max() - coor.min())) - 1 

        # define sin wave freq
        freqs = torch.arange(D, dtype=torch.float).cuda() 
        freqs = np.pi * (2**freqs)       

        freqs = freqs.view(*[1]*len(normal_coor.shape), -1) # 1 x 1 x 1 x D
        normal_coor = normal_coor.unsqueeze(-1) # B x 3 x N x 1
        k = normal_coor * freqs # B x 3 x N x D
        s = torch.sin(k) # B x 3 x N x D
        c = torch.cos(k) # B x 3 x N x D
        x = torch.cat([s,c], -1) # B x 3 x N x 2D
        pos = x.transpose(-1,-2).reshape(coor.shape[0], -1, coor.shape[-1]) # B 6D N
        # zero_pad = torch.zeros(x.size(0), 2, x.size(-1)).cuda()
        # pos = torch.cat([x, zero_pad], dim = 1)
        # pos = self.pos_embed_wave(x)
        return pos

    def forward(self, inpc):
        '''
            inpc : input incomplete point cloud with shape B N(2048) C(3)
        '''
        # build point proxy
        bs = inpc.size(0)
        points = self.grouper(inpc.transpose(1,2).contiguous()) # points: bs, [3 + 256], num_points // 16
        coords = points[:, :3, :] # bs, 3, num_points // 16
        features = points[:, 3:, :] # bs, 256, num_points // 16
        
        pos_embed =  self.pos_embed(coords) # pos_embed: bs, embed_dim, num_points // 16
        input_features = self.input_proj(features) # x: bs, embed_dim, num_points // 16

        encoder_input = torch.cat([coords, pos_embed + input_features], dim=1)

        # encoder
        encoder_output = self.encoder(encoder_input) # bs, 3 + embed_dim, num_points // 16


        # Query Generator
        global_feature = encoder_output[:, 3:, :]
        global_feature = self.increase_dim(global_feature) # bs, 1024, num_points // 16
        global_feature = torch.max(global_feature, dim=-1)[0] # bs 1024

        # Coarse Point Cloud
        coarse_point_cloud = self.coarse_pred(global_feature).reshape(bs, -1, 3)  #  bs, num_query, 3

        query_feature = torch.cat([coarse_point_cloud,global_feature.unsqueeze(1).expand(-1, self.num_query, -1)], dim=-1) # bs, num_query, 3 + embed_dim
        query_feature = self.query_conv(query_feature.transpose(1,2)).transpose(1,2) # bs, num_query, embed_dim
        
        key_points = encoder_output
        query_points = torch.cat([coarse_point_cloud, query_feature], dim=-1).transpose(1,2) # bs, 3 + embed_dim, num_query
        
        # decoder
        points = self.decoder(query_points, key_points)

        return points

if __name__ == "__main__":
    x = torch.randn(64, 2048, 3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    model = PCTransformer(in_chans=3, embed_dim=384, depth=[[1, 5], [1, 5]], num_heads=6, num_query = 224).to(device)
    points = model(x)
    print(points.shape)