"""
PCTransformer model package.

This package contains the modular implementation of the PCTransformer
for point cloud completion tasks.
"""

# Main model classes
from .PoinTrPCTransformer import PoinTrPCTransformer
from .PoinTr import PoinTr
from .AdaPoinTr import AdaPoinTr

# Attention and transformer components
from .Attention import MultiHeadAttention, FeedForward
from .TransformerBlocks import (
    SelfAttentionBlock,
    GeometryAwareSelfAttentionBlock,
    CrossAttentionBlock,
    GeometryAwareCrossAttentionBlock
)

# Encoder and Decoder components
from .GAEncoder import GeometryAwareTransformerEncoder
from .GADecoder import GeometryAwareTransformerDecoder

# Query generation and utility components
from .QueryGenerator import DynamicQueryGenerator, AdaptiveDenoisingQueryGenerator
from .DGCNN import kNNQuery, EdgeConv, DGCNN_Grouper

# Utility functions
from .Utils import (
    fps_downsample,
    knn_index,
    square_distance,
    sinusoidal_position_encoding,
    extract_coordinates_and_features,
    combine_coordinates_and_features
)

__all__ = [
    # Main models
    'PoinTrPCTransformer',
    'PoinTr',
    'AdaPoinTr',
    
    # Attention components
    'MultiHeadAttention',
    'FeedForward',
    'SelfAttentionBlock',
    'GeometryAwareSelfAttentionBlock',
    'CrossAttentionBlock',
    'GeometryAwareCrossAttentionBlock',
    
    # Encoder/Decoder components
    'GeometryAwareTransformerEncoder',
    'GeometryAwareTransformerDecoder',
    
    # Query and grouping components
    'DynamicQueryGenerator',
    'AdaptiveDenoisingQueryGenerator',
    'kNNQuery',
    'EdgeConv',
    'DGCNN_Grouper',
    
    # Utility functions
    'fps_downsample',
    'knn_index',
    'square_distance',
    'sinusoidal_position_encoding',
    'extract_coordinates_and_features',
    'combine_coordinates_and_features'
] 