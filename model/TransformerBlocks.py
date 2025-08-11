import torch
import torch.nn as nn

from model.Attention import MultiHeadAttention, FeedForward, GraphAttention


class SelfAttentionBlock(nn.Module):
    """
    Standard self-attention transformer block.
    
    Implements a vanilla transformer block with self-attention and
    feed-forward network with residual connections.
    """
    def __init__(self, d_model=384, num_heads=6):
        """
        Initialize the self-attention block.
        
        Args:
            d_model (int): Feature dimension
            num_heads (int): Number of attention heads
        """
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads
        )
        self.feed_forward = FeedForward(d_model, hidden_channels=d_model * 2)
        self.feed_forward_norm = nn.LayerNorm(d_model)
    
    def forward(self, features, mask=None):
        """
        Forward pass for vanilla transformer block.
        
        Args:
            points (torch.Tensor): Input points with shape [batch, 3+feature_dim, num_points]
            
        Returns:
            torch.Tensor: Output points with shape [batch, 3+feature_dim, num_points]
        """
        # Normalize features
        features = features.transpose(1, 2).contiguous()
        norm_features = self.input_norm(features)  # [batch, num_points, feature_dim]
        
        # Self-attention
        qkv = self.qkv_proj(norm_features)  # [batch, num_points, 3*feature_dim]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch, num_points, feature_dim]
        attn_features = self.multi_head_attention(q, k, v, mask=mask)  # [batch, num_points, feature_dim]
        
        # Residual connection
        features = features + attn_features  # [batch, num_points, feature_dim]
        
        # Feed-forward network
        features = self.feed_forward(self.feed_forward_norm(features)) + features
        
        return features.transpose(1, 2)


class GeometryAwareSelfAttentionBlock(nn.Module):
    """
    Geometry-aware self-attention transformer block.
    
    Extends the standard self-attention block with geometry-aware features
    computed using k-nearest neighbors.
    """
    def __init__(self, d_model=384, num_heads=6, k_neighbors=8):
        """
        Initialize the geometry-aware self-attention block.
        
        Args:
            d_model (int): Feature dimension
            num_heads (int): Number of attention heads
            k_neighbors (int): Number of k-nearest neighbors for geometry features
        """
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads
        )
        self.feed_forward = FeedForward(d_model, hidden_channels=d_model * 2)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        
        # Geometry-aware components
        self.graph_attention = GraphAttention(d_model, k_neighbors=k_neighbors)
        
        # Merge attention and geometry features
        self.merge_proj = nn.Linear(d_model * 2, d_model)
        
    def forward(self, coords, features, mask=None):
        """
        Forward pass for geometry-aware transformer block.
        
        Args:
            points (torch.Tensor): Input points with shape [batch, 3+feature_dim, num_points]
            
        Returns:
            torch.Tensor: Output points with shape [batch, 3+feature_dim, num_points]
        """
        # Normalize features
        features = features.transpose(1, 2).contiguous()
        norm_features = self.input_norm(features)  # [batch, num_points, feature_dim]
        
        # Self-attention
        qkv = self.qkv_proj(norm_features)  # [batch, num_points, 3*feature_dim]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch, num_points, feature_dim]
        attn_features = self.multi_head_attention(q, k, v, mask=mask)  # [batch, num_points, feature_dim]
        
        # Geometry-aware features using kNN
        # Get geometry features
        geom_features = self.graph_attention(coords, norm_features.transpose(1, 2), coords, norm_features.transpose(1, 2))  # [batch, num_points, feature_dim]
        
        # Merge attention and geometry features
        attn_features = torch.cat([attn_features, geom_features], dim=-1)  # [batch, num_points, 2*feature_dim]
        attn_features = self.merge_proj(attn_features)  # [batch, num_points, feature_dim]
        
        # Residual connection
        features = features + attn_features  # [batch, num_points, feature_dim]
        
        # Feed-forward network
        features = features + self.feed_forward(self.feed_forward_norm(features))
        
        return features.transpose(1, 2)


class CrossAttentionBlock(nn.Module):
    """
    Standard cross-attention transformer block.
    
    Implements cross-attention between query and key points with
    separate normalization and projection layers.
    """
    def __init__(self, d_model=384, num_heads=6, attn_drop=0., proj_drop=0.):
        """
        Initialize the cross-attention block.
        
        Args:
            d_model (int): Feature dimension
            num_heads (int): Number of attention heads
            attn_drop (float): Attention dropout rate
            proj_drop (float): Projection dropout rate
        """
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads
        )
        self.cross_multi_head_attention = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads
        )
        self.feed_forward = FeedForward(d_model, hidden_channels=d_model * 2)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        
        # Cross-attention components
        self.cross_norm_q = nn.LayerNorm(d_model)
        self.cross_norm_k = nn.LayerNorm(d_model)
        self.cross_q_map = nn.Linear(d_model, d_model, bias=False)
        self.cross_k_map = nn.Linear(d_model, d_model, bias=False)
        self.cross_v_map = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, query_features, key_features, mask=None):
        """
        Forward pass for cross-attention block.
        
        Args:
            query_points (torch.Tensor): Query points with shape [batch, 3+feature_dim, num_query]
            key_points (torch.Tensor): Key points with shape [batch, 3+feature_dim, num_key]
            
        Returns:
            torch.Tensor: Updated query points with shape [batch, 3+feature_dim, num_query]
        """
        # Normalize features
        query_features = query_features.transpose(1, 2).contiguous()
        norm_features = self.input_norm(query_features)  # [batch, num_points, feature_dim]
        
        # Self-attention
        qkv = self.qkv_proj(norm_features)  # [batch, num_points, 3*feature_dim]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch, num_points, feature_dim]
        attn_features = self.multi_head_attention(q, k, v, mask=mask)  # [batch, num_points, feature_dim]
        
        query_features = query_features + attn_features
        
        # Cross-attention
        normed_query_features = self.cross_norm_q(query_features)
        normed_key_features = self.cross_norm_k(key_features.transpose(1,2))
        cross_q = self.cross_q_map(normed_query_features)
        cross_k = self.cross_k_map(normed_key_features)
        cross_v = self.cross_v_map(normed_key_features)
        cross_features = self.cross_multi_head_attention(cross_q, cross_k, cross_v)
        
        query_features = query_features + cross_features
        query_features = query_features + self.feed_forward(self.feed_forward_norm(query_features))
        
        return query_features.transpose(1,2)


class GeometryAwareCrossAttentionBlock(nn.Module):
    """
    Geometry-aware cross-attention transformer block.
    
    Combines self-attention and cross-attention with geometry-aware features
    computed using k-nearest neighbors for both self and cross attention.
    """
    def __init__(self, d_model=384, num_heads=6, k_neighbors=8):
        """
        Initialize the geometry-aware cross-attention block.
        
        Args:
            d_model (int): Feature dimension
            num_heads (int): Number of attention heads
            k_neighbors (int): Number of k-nearest neighbors for geometry features
        """
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads
        )
        self.cross_multi_head_attention = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads
        )
        self.feed_forward = FeedForward(d_model, hidden_channels=d_model * 2)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        
        # Geometry-aware components
        self.self_graph_attention = GraphAttention(d_model, k_neighbors=k_neighbors)
        self.cross_graph_attention = GraphAttention(d_model, k_neighbors=k_neighbors)
        
        self.self_merge_proj = nn.Linear(d_model * 2, d_model)
        self.cross_merge_proj = nn.Linear(d_model * 2, d_model)
        
        # Cross-attention components
        self.cross_norm_q = nn.LayerNorm(d_model)
        self.cross_norm_k = nn.LayerNorm(d_model)
        self.cross_q_map = nn.Linear(d_model, d_model, bias=False)
        self.cross_k_map = nn.Linear(d_model, d_model, bias=False)
        self.cross_v_map = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, query_coords, query_features, key_coords, key_features, mask=None):
        """
        Forward pass for geometry-aware cross-attention block.
        
        Args:
            query_points (torch.Tensor): Query points with shape [batch, 3+feature_dim, num_query]
            key_points (torch.Tensor): Key points with shape [batch, 3+feature_dim, num_key]
            
        Returns:
            torch.Tensor: Updated query points with shape [batch, 3+feature_dim, num_query]
        """
        #################################
        # Geometry-aware Self-attention #
        #################################
        # Normalize features
        query_features = query_features.transpose(1, 2).contiguous()
        norm_features = self.input_norm(query_features)  # [batch, num_points, feature_dim]
        
        # Self-attention
        qkv = self.qkv_proj(norm_features)  # [batch, num_points, 3*feature_dim]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch, num_points, feature_dim]
        attn_features = self.multi_head_attention(q, k, v, mask=mask)  # [batch, num_points, feature_dim]
        
        # Geometry-aware features using kNN
        # Get geometry features
        geom_features = self.self_graph_attention(query_coords, norm_features.transpose(1, 2), query_coords, norm_features.transpose(1, 2))  # [batch, num_points, feature_dim]
        
        # Merge attention and geometry features
        attn_features = torch.cat([attn_features, geom_features], dim=-1)  # [batch, num_points, 2*feature_dim]
        attn_features = self.self_merge_proj(attn_features)  # [batch, num_points, feature_dim]
        
        # Residual connection
        query_features = query_features + attn_features  # [batch, num_points, feature_dim]
        
        ##################################
        # Geometry-aware Cross-attention #
        ##################################
        
        normed_query_features = self.cross_norm_q(query_features)
        normed_key_features = self.cross_norm_k(key_features.transpose(1,2))
        
        cross_q = self.cross_q_map(normed_query_features)
        cross_k = self.cross_k_map(normed_key_features)
        cross_v = self.cross_v_map(normed_key_features)
        cross_features = self.cross_multi_head_attention(cross_q, cross_k, cross_v)
        
        cross_geom_features = self.cross_graph_attention(query_coords, normed_query_features.transpose(1,2), key_coords, normed_key_features.transpose(1,2))  # [batch, num_points, feature_dim]
        
        cross_features = torch.cat([cross_features, cross_geom_features], dim=-1)  # [batch, num_points, 2*feature_dim]
        cross_features = self.cross_merge_proj(cross_features)  # [batch, num_points, feature_dim]
        
        query_features = query_features + cross_features
        query_features = query_features + self.feed_forward(self.feed_forward_norm(query_features))
        
        return query_features.transpose(1, 2)