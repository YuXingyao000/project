"""
Encoder and decoder components for the PCTransformer.

This module contains the encoder and decoder architectures that
combine multiple transformer blocks for point cloud processing.
"""

import torch.nn as nn

from transformer_blocks import (
    SelfAttentionBlock, 
    GeometryAwareSelfAttentionBlock,
    CrossAttentionBlock,
    GeometryAwareCrossAttentionBlock
)


class GeometryAwareTransformerEncoder(nn.Module):
    """
    PCTransformer Encoder with geometry-aware attention blocks.
    
    Combines geometry-aware and vanilla self-attention blocks to process
    point cloud features with both geometric and semantic information.
    """
    def __init__(self, d_model=384, geom_block_num=1, vanilla_block_num=5, num_heads=6):
        """
        Initialize the geometry-aware transformer encoder.
        
        Args:
            d_model (int): Feature dimension
            geom_block_num (int): Number of geometry-aware blocks
            vanilla_block_num (int): Number of vanilla transformer blocks
            num_heads (int): Number of attention heads
        """
        super().__init__()
        assert geom_block_num + vanilla_block_num > 0, "Block number must be greater than 0"
        
        self.blocks = nn.ModuleList([
            GeometryAwareSelfAttentionBlock(d_model=d_model, num_heads=num_heads) 
            for _ in range(geom_block_num)
        ])
        self.blocks.extend([
            SelfAttentionBlock(d_model=d_model, num_heads=num_heads) 
            for _ in range(vanilla_block_num)
        ])
    
    def forward(self, points):
        """
        Forward pass through the encoder.
        
        Args:
            points (torch.Tensor): Input points with shape [batch, 3+feature_dim, num_points]
            
        Returns:
            torch.Tensor: Encoded points with shape [batch, 3+feature_dim, num_points]
        """
        for block in self.blocks:
            points = block(points)
        return points


class GeometryAwareTransformerDecoder(nn.Module):
    """
    PCTransformer Decoder with geometry-aware cross-attention blocks.
    
    Combines geometry-aware and vanilla cross-attention blocks to generate
    point cloud completions from encoded features.
    """
    def __init__(self, d_model=384, geom_block_num=1, cross_block_num=5, num_heads=6):
        """
        Initialize the geometry-aware transformer decoder.
        
        Args:
            d_model (int): Feature dimension
            geom_block_num (int): Number of geometry-aware cross-attention blocks
            cross_block_num (int): Number of vanilla cross-attention blocks
            num_heads (int): Number of attention heads
        """
        super().__init__()
        self.blocks = nn.ModuleList([
            GeometryAwareCrossAttentionBlock(d_model=d_model, num_heads=num_heads) 
            for _ in range(geom_block_num)
        ])
        self.blocks.extend([
            CrossAttentionBlock(d_model=d_model, num_heads=num_heads) 
            for _ in range(cross_block_num)
        ])
    
    def forward(self, query_points, key_points):
        """
        Forward pass through the decoder.
        
        Args:
            query_points (torch.Tensor): Query points with shape [batch, 3+feature_dim, num_query]
            key_points (torch.Tensor): Key points with shape [batch, 3+feature_dim, num_key]
            
        Returns:
            torch.Tensor: Decoded query points with shape [batch, 3+feature_dim, num_query]
        """
        for block in self.blocks:
            query_points = block(query_points, key_points)
        return query_points 