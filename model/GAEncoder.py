import torch
import torch.nn as nn

from model.DGCNN import DGCNN_Grouper
from model.TransformerBlocks import (
    SelfAttentionBlock, 
    GeometryAwareSelfAttentionBlock,
)

class GeometryAwareTransformerEncoder(nn.Module):
    def __init__(self, in_chans=3, embed_dim=384, num_heads=6, depth=[1, 5], grouper_downsample=[4, 16], grouper_k_nearest_neighbors=16, attention_k_nearest_neighbors=8, norm_eps=1e-5):
        """
        Args:
            - in_chans (int): Number of input channels (coordinates)
            - embed_dim (int): Embedding dimension for features
            - num_heads (int): Number of attention heads
            - depth (list): List of [geom_blocks, vanilla_blocks]
            - grouper_downsample (list): downsample divisor for the grouper to generate input proxy
            - grouper_k_nearest_neighbors (int): Number of nearest neighbors to use for the grouper
            - attention_k_nearest_neighbors (int): Number of nearest neighbors to use for attention blocks
            - norm_eps (float): Epsilon for layer normalization
        """
        super().__init__()
        
        assert len(depth) == 2, f"depth must be a list of two elements, got {depth}"
        assert len(grouper_downsample) == 2, f"grouper_downsample must be a list of two elements, got {grouper_downsample}"
        
        self.embed_dim = embed_dim
        
        self.geom_depth = depth[0]
        
        # Point cloud grouping and feature extraction
        self.grouper = DGCNN_Grouper(
            input_dim=in_chans, 
            grouper_downsample=grouper_downsample, 
            k_nearest_neighbors=grouper_k_nearest_neighbors
            )  # B 3 N to B C(3) N(128) and B C(128) N(128)

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
            [GeometryAwareSelfAttentionBlock(d_model=embed_dim, num_heads=num_heads, k_nearest_neighbors=attention_k_nearest_neighbors, norm_eps=norm_eps) for _ in range(depth[0])] +
            [SelfAttentionBlock(d_model=embed_dim, num_heads=num_heads, norm_eps=norm_eps) for _ in range(depth[1])]
        )

    def _build_input_proxy(self, incomplete_point_cloud):
        """
        Build point input proxy from incomplete point cloud.
        
        Args:
            - incomplete_point_cloud (torch.Tensor): Input incomplete point cloud [batch, num_points, 3]
            
        Returns:
            - coordinates: [batch, 3, num_points//[grouper_downsample[0]]]
            - features: [batch, grouper_feature_dim, num_points//grouper_downsample[1]]
        """
        # Group points and extract features
        coords, features = self.grouper(incomplete_point_cloud.transpose(1, 2).contiguous())
        
        return coords, features

    def _encode_input_proxy(self, coords, features):
        """
        Encode point features with position embedding.
        
        Args:
            - coords (torch.Tensor): Point coordinates [batch, 3, num_points]
            - features (torch.Tensor): Point features [batch, feature_dim, num_points]
            
        Returns:
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
            - incomplete_point_cloud (torch.Tensor): Input incomplete point cloud [batch, num_points, 3]
            
        Returns:
            - coords: [batch, num_points//grouper_downsample[1], 3]
            - encoded_features: [batch, num_points//grouper_downsample[1], embed_dim]
        """
        # Build point proxy from the partial point cloud
        coords, features = self._build_input_proxy(incomplete_point_cloud)
        
        # Encode input proxy
        pos_embed, x = self._encode_input_proxy(coords, features)
        
        # API fitting
        pos_embed = pos_embed.transpose(1, 2)
        x = x.transpose(1, 2)
        coords = coords.transpose(1, 2)
        
        # Geometry-aware Transformer Encoder
        for i, encoder_block in enumerate(self.encoder):
            if i < self.geom_depth:
                x = encoder_block(coords, x + pos_embed)
            else:
                x = encoder_block(x + pos_embed)
        
        return coords, x