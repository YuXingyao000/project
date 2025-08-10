"""
PoinTrPCTransformer: Refactored Point Cloud Transformer for completion tasks.

This module implements the PCTransformer architecture divided into three main components:
1. Geometry-aware Transformer Encoder
2. Query Generator  
3. Geometry-aware Transformer Decoder

These components are then composed together in the main PoinTrPCTransformer class.
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from model.DGCNN import DGCNN_Grouper
from model.TransformerBlocks import (
    SelfAttentionBlock, 
    GeometryAwareSelfAttentionBlock,
    CrossAttentionBlock,
    GeometryAwareCrossAttentionBlock
)
from model.QueryGenerator import QueryGenerator


class GeometryAwareTransformerEncoder(nn.Module):
    """
    Geometry-aware Transformer Encoder component.
    
    This component processes the incomplete point cloud through:
    1. Point proxy building using DGCNN grouper
    2. Feature encoding with position embeddings
    3. Geometry-aware and vanilla transformer blocks
    """
    
    def __init__(self, in_chans=3, embed_dim=384, depth=[1, 5], num_heads=6):
        """
        Initialize the Geometry-aware Transformer Encoder.
        
        Args:
            in_chans (int): Number of input channels (coordinates)
            embed_dim (int): Embedding dimension for features
            depth (list): List of [geom_blocks, vanilla_blocks]
            num_heads (int): Number of attention heads
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
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
            _, x = encoder_block(torch.cat([coords, x + pos_embed], dim=1))
        
        return coords, x





class GeometryAwareTransformerDecoder(nn.Module):
    """
    Geometry-aware Transformer Decoder component.
    
    This component refines the query points through:
    1. Geometry-aware cross-attention between queries and encoded features
    2. Vanilla cross-attention blocks
    """
    
    def __init__(self, embed_dim=384, depth=[1, 7], num_heads=6):
        """
        Initialize the Geometry-aware Transformer Decoder.
        
        Args:
            embed_dim (int): Embedding dimension for features
            depth (list): List of [geom_blocks, vanilla_blocks]
            num_heads (int): Number of attention heads
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Transformer decoder blocks
        self.decoder = nn.ModuleList(
            [GeometryAwareCrossAttentionBlock(d_model=embed_dim, num_heads=num_heads) for _ in range(depth[0])] +
            [CrossAttentionBlock(d_model=embed_dim, num_heads=num_heads) for _ in range(depth[1])]
        )

    def forward(self, query_coordinate, query_feature, key_coordinate, key_feature):
        """
        Forward pass of the Geometry-aware Transformer Decoder.
        
        Args:
            query_coordinate (torch.Tensor): Query coordinates [batch, 3, num_query]
            query_feature (torch.Tensor): Query features [batch, embed_dim, num_query]
            key_coordinate (torch.Tensor): Key coordinates [batch, 3, num_points]
            key_feature (torch.Tensor): Key features [batch, embed_dim, num_points]
            
        Returns:
            torch.Tensor: Refined query features [batch, embed_dim, num_query]
        """
        # Geometry-aware Transformer Decoder
        for i, decoder_block in enumerate(self.decoder):
            _, query_feature = decoder_block(
                torch.cat([query_coordinate, query_feature], dim=1), 
                torch.cat([key_coordinate, key_feature], dim=1)
            )
        
        return query_feature


class PoinTrPCTransformer(nn.Module):
    """
    PoinTrPCTransformer: Composed Point Cloud Transformer for completion tasks.
    
    This model combines three main components:
    1. GeometryAwareTransformerEncoder: Processes incomplete point cloud
    2. QueryGenerator: Generates query points for completion
    3. GeometryAwareTransformerDecoder: Refines query points
    """
    
    def __init__(self, in_chans=3, embed_dim=384, depth=[[1, 5], [1, 7]], num_heads=6, num_query=224):
        """
        Initialize the PoinTrPCTransformer model.
        
        Args:
            in_chans (int): Number of input channels (coordinates)
            embed_dim (int): Embedding dimension for features
            depth (list): List of [encoder_depth, decoder_depth] where each depth is [geom_blocks, vanilla_blocks]
            num_heads (int): Number of attention heads
            num_query (int): Number of query points for completion
        """
        super().__init__()

        self.num_features = self.embed_dim = embed_dim
        
        # Initialize the three main components
        self.encoder = GeometryAwareTransformerEncoder(
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth[0],
            num_heads=num_heads
        )
        
        self.query_generator = QueryGenerator(
            embed_dim=embed_dim,
            num_query=num_query
        )
        
        self.decoder = GeometryAwareTransformerDecoder(
            embed_dim=embed_dim,
            depth=depth[1],
            num_heads=num_heads
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

    def forward(self, incomplete_point_cloud):
        """
        Forward pass of the PoinTrPCTransformer.
        
        Args:
            incomplete_point_cloud (torch.Tensor): Input incomplete point cloud with shape [batch, num_points, 3]
            
        Returns:
            tuple: (query_features, coarse_point_cloud) where:
                - query_features: [batch, num_query, embed_dim]
                - coarse_point_cloud: [batch, num_query, 3]
        """
        # Step 1: Geometry-aware Transformer Encoder
        coords, encoded_features = self.encoder(incomplete_point_cloud)
        
        # Step 2: Query Generator
        coarse_point_cloud, query_feature = self.query_generator(encoded_features)
        
        # Step 3: Geometry-aware Transformer Decoder
        refined_query_feature = self.decoder(
            query_coordinate=coarse_point_cloud.transpose(1, 2),  # [batch, 3, num_query]
            query_feature=query_feature,                          # [batch, embed_dim, num_query]
            key_coordinate=coords,                                # [batch, 3, num_points//16]
            key_feature=encoded_features                          # [batch, embed_dim, num_points//16]
        )
        
        # Return the final results
        return refined_query_feature.transpose(1, 2).contiguous(), coarse_point_cloud


if __name__ == "__main__":
    # Test the refactored model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test input
    batch_size = 2
    num_points = 2048
    test_input = torch.randn(batch_size, num_points, 3).to(device)
    
    # Create model
    model = PoinTrPCTransformer(
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
        
        # Test individual components
        print("\nTesting individual components:")
        coords, encoded_features = model.encoder(test_input)
        print(f"Encoder output - coords: {coords.shape}, encoded_features: {encoded_features.shape}")
        
        coarse_points, query_feat = model.query_generator(encoded_features)
        print(f"Query Generator output - coarse_points: {coarse_points.shape}, query_feat: {query_feat.shape}")
        
        refined_query = model.decoder(
            query_coordinate=coarse_points.transpose(1, 2),
            query_feature=query_feat,
            key_coordinate=coords,
            key_feature=encoded_features
        )
        print(f"Decoder output - refined_query: {refined_query.shape}") 