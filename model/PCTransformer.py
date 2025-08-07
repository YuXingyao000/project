"""
PCTransformer: Point Cloud Transformer for completion tasks.

This module implements the main PCTransformer architecture that combines
geometry-aware attention mechanisms with transformer blocks for point cloud completion.
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from model.DGCNN import DGCNN_Grouper
from model.transformer_blocks import (
    SelfAttentionBlock, 
    GeometryAwareSelfAttentionBlock,
    CrossAttentionBlock,
    GeometryAwareCrossAttentionBlock
)


class PCTransformer(nn.Module):
    """
    PCTransformer: Vision Transformer with support for point cloud completion.
    
    This model uses a combination of geometry-aware and vanilla transformer blocks
    to process incomplete point clouds and generate completions.
    """
    
    def __init__(self, in_chans=3, embed_dim=384, depth=[[1, 5], [1, 5]], num_heads=6, num_query=224):
        """
        Initialize the PCTransformer model.
        
        Args:
            in_chans (int): Number of input channels (coordinates)
            embed_dim (int): Embedding dimension for features
            depth (list): List of [encoder_depth, decoder_depth] where each depth is [geom_blocks, vanilla_blocks]
            num_heads (int): Number of attention heads
            num_query (int): Number of query points for completion
        """
        super().__init__()

        self.num_features = self.embed_dim = embed_dim
        
        # Point cloud grouping and feature extraction
        self.grouper = DGCNN_Grouper()  # B 3 N to B C(3) N(128) and B C(128) N(128)

        # Position embedding network
        self.pos_embed = nn.Sequential(
            nn.Conv1d(in_chans, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, embed_dim, 1)
        )

        # Input feature projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(128, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(embed_dim, embed_dim, 1)
        )

        # Transformer encoder
        self.encoder = nn.ModuleList(
            [GeometryAwareSelfAttentionBlock(d_model=embed_dim, num_heads=num_heads) for _ in range(depth[0][0])] +
            [SelfAttentionBlock(d_model=embed_dim, num_heads=num_heads) for _ in range(depth[0][1])]
        )

        # Feature dimension increase for global representation
        self.increase_dim = nn.Sequential(
            nn.Conv1d(embed_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        # Query generation components
        self.num_query = num_query
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * num_query)
        )
        
        # Query feature processing
        self.query_conv = nn.Sequential(
            nn.Conv1d(1024 + 3, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, embed_dim, 1)
        )

        # Transformer decoder
        self.decoder = nn.ModuleList(
            [GeometryAwareCrossAttentionBlock(d_model=embed_dim, num_heads=num_heads) for _ in range(depth[1][0])] +
            [CrossAttentionBlock(d_model=embed_dim, num_heads=num_heads) for _ in range(depth[1][1])]
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize model weights using appropriate initialization strategies.
        
        Args:
            m (nn.Module): Module to initialize
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def _build_point_proxy(self, incomplete_point_cloud):
        """
        Build point proxy from incomplete point cloud.
        
        Args:
            incomplete_point_cloud (torch.Tensor): Input incomplete point cloud [batch, num_points, 3]
            
        Returns:
            tuple: (coordinates, features) where:
                - coordinates: [batch, 3, num_points//16]
                - features: [batch, 256, num_points//16]
        """
        # Group points and extract features
        coords, features = self.grouper(incomplete_point_cloud.transpose(1, 2).contiguous())
        
        return coords, features

    def _encode_point_features(self, coords, features):
        """
        Encode point features with position embedding.
        
        Args:
            coords (torch.Tensor): Point coordinates [batch, 3, num_points]
            features (torch.Tensor): Point features [batch, feature_dim, num_points]
            
        Returns:
            torch.Tensor: Encoded features [batch, embed_dim, num_points]
        """
        # Generate position embedding
        pos_embed = self.pos_embed(coords)  # [batch, embed_dim, num_points]
        
        # Project input features
        input_features = self.input_proj(features)  # [batch, embed_dim, num_points]
        
        return pos_embed, input_features

    def _generate_query_points(self, global_feature):
        """
        Generate query points for completion.
        
        Args:
            global_feature (torch.Tensor): Global feature vector [batch, 1024]
            
        Returns:
            tuple: (coarse_points, query_features) where:
                - coarse_points: [batch, num_query, 3]
                - query_features: [batch, embed_dim, num_query]
        """
        # Generate coarse point cloud
        coarse_point_cloud = self.coarse_pred(global_feature).reshape(-1, self.num_query, 3)
        
        # Prepare query features
        query_feature = torch.cat([
            global_feature.unsqueeze(1).expand(-1, self.num_query, -1),
            coarse_point_cloud
        ], dim=-1)  # [batch, num_query, 3 + 1024]
        
        # Process query features
        query_feature = self.query_conv(query_feature.transpose(1, 2)).transpose(1, 2)
        
        return coarse_point_cloud, query_feature

    def forward(self, incomplete_point_cloud):
        """
        Forward pass of the PCTransformer.
        
        Args:
            incomplete_point_cloud (torch.Tensor): Input incomplete point cloud with shape [batch, num_points, 3]
            
        Returns:
            torch.Tensor: Completed point cloud with shape [batch, 3+embed_dim, num_query]
        """
        batch_size = incomplete_point_cloud.size(0)
        
        # Step 1: Build point proxy
        coords, features = self._build_point_proxy(incomplete_point_cloud)
        
        # Step 2: Encode point features
        pos_embed, x = self._encode_point_features(coords, features)
        
        # TODO: Need to be modified, the encoder is not correct
        for i, encoder_block in enumerate(self.encoder):
            _, x = encoder_block(torch.cat([coords, x + pos_embed], dim=1))
        
        # Step 5: Generate global feature
        global_feature = self.increase_dim(x)  # [batch, 1024, num_points//16]
        global_feature = torch.max(global_feature, dim=-1)[0]  # [batch, 1024]
        
        # TODO: Not correct, coarse_point_cloud is not implemented
        # Generate coarse point cloud
        coarse_point_cloud = self.coarse_pred(global_feature).reshape(-1, self.num_query, 3)
        # Prepare query features
        query_feature = torch.cat([
            global_feature.unsqueeze(1).expand(-1, self.num_query, -1),
            coarse_point_cloud
        ], dim=-1)
        # Process query features
        query_feature = self.query_conv(query_feature.transpose(1, 2))
        
        for i, decoder_block in enumerate(self.decoder):
            _, query_feature = decoder_block(torch.cat([coarse_point_cloud.transpose(1, 2), query_feature], dim=1), torch.cat([coords, x], dim=1))

        return query_feature.transpose(1, 2).contiguous(), coarse_point_cloud


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test input
    batch_size = 2
    num_points = 2048
    test_input = torch.randn(batch_size, num_points, 3).to(device)
    
    # Create model
    model = PCTransformer(
        in_chans=3, 
        embed_dim=384, 
        depth=[[1, 5], [1, 7]], 
        num_heads=6, 
        num_query=224
    ).to(device)
    
    # Test forward pass
    with torch.no_grad():
        query_features, coarse_point_cloud = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Query features shape: {query_features.shape}")
        print(f"Coarse point cloud shape: {coarse_point_cloud.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")