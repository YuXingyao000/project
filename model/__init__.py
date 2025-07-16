"""
PCTransformer model package.

This package contains the modular implementation of the PCTransformer
for point cloud completion tasks.
"""

from .PCTransformer import PCTransformer
from .attention import MultiHeadAttention, FeedForward
from .transformer_blocks import (
    SelfAttentionBlock,
    GeometryAwareSelfAttentionBlock,
    CrossAttentionBlock,
    GeometryAwareCrossAttentionBlock
)
from .encoder import GeometryAwareTransformerEncoder, GeometryAwareTransformerDecoder
from .geometry import (
    sinusoidal_position_encoding,
    extract_coordinates_and_features,
    combine_coordinates_and_features
)
from .PoinTr import PoinTr

__all__ = [
    'PCTransformer',
    'MultiHeadAttention',
    'FeedForward',
    'SelfAttentionBlock',
    'GeometryAwareSelfAttentionBlock',
    'CrossAttentionBlock',
    'GeometryAwareCrossAttentionBlock',
    'GeometryAwareTransformerEncoder',
    'GeometryAwareTransformerDecoder',
    'sinusoidal_position_encoding',
    'extract_coordinates_and_features',
    'combine_coordinates_and_features'
] 