import torch
import torch.nn as nn

from model.DGCNN import DGCNN_Grouper
from model.TransformerBlocks import (
    SelfAttentionBlock, 
    GeometryAwareSelfAttentionBlock,
)

class GeometryAwareTransformerEncoder(nn.Module):
    def __init__(self, in_chans=3, embed_dim=384, depth=[1, 5], num_heads=6):
        """
        Args:
            - in_chans (int): Number of input channels (coordinates)
            - embed_dim (int): Embedding dimension for features
            - depth (list): List of [geom_blocks, vanilla_blocks]
            - num_heads (int): Number of attention heads
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        self.geom_depth = depth[0]
        
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

        # Transformer encoder blocks
        self.encoder = nn.ModuleList(
            [GeometryAwareSelfAttentionBlock(d_model=embed_dim, num_heads=num_heads) for _ in range(depth[0])] +
            [SelfAttentionBlock(d_model=embed_dim, num_heads=num_heads) for _ in range(depth[1])]
        )

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
            tuple: (pos_embed, input_features) where:
                - pos_embed: [batch, embed_dim, num_points]
                - input_features: [batch, embed_dim, num_points]
        """
        # Generate position embedding
        pos_embed = self.pos_embed(coords)  # [batch, embed_dim, num_points]
        
        # Project input features
        input_features = self.input_proj(features)  # [batch, embed_dim, num_points]
        
        return pos_embed, input_features

    def forward(self, incomplete_point_cloud):
        """
        Forward pass of the Geometry-aware Transformer Encoder.
        
        Args:
            incomplete_point_cloud (torch.Tensor): Input incomplete point cloud [batch, num_points, 3]
            
        Returns:
            tuple: (coords, encoded_features) where:
                - coords: [batch, 3, num_points//16]
                - encoded_features: [batch, embed_dim, num_points//16]
        """
        # Build point proxy from the partial point cloud
        coords, features = self._build_point_proxy(incomplete_point_cloud)
        
        # Encode point features with position embedding
        pos_embed, x = self._encode_point_features(coords, features)
        
        # Geometry-aware Transformer Encoder
        for i, encoder_block in enumerate(self.encoder):
            if i < self.geom_depth:
                x = encoder_block(coords, x + pos_embed)
            else:
                x = encoder_block(x + pos_embed)
        
        return coords, x