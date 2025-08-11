import torch
import torch.nn as nn

from model.TransformerBlocks import (
    CrossAttentionBlock,
    GeometryAwareCrossAttentionBlock
)

class GeometryAwareTransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=[1, 7], num_heads=6):
        """
        Args:
            - embed_dim (int): Embedding dimension for features
            - depth (list): List of [geom_blocks, vanilla_blocks]
            - num_heads (int): Number of attention heads
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
        Args:
            query_coordinate (torch.Tensor): Query coordinates [batch, 3, num_query]
            query_feature (torch.Tensor): Query features [batch, embed_dim, num_query]
            key_coordinate (torch.Tensor): Key coordinates [batch, 3, num_points]
            key_feature (torch.Tensor): Key features [batch, embed_dim, num_points]
            
        Returns:
            torch.Tensor: Query features [batch, embed_dim, num_query]
        """
        # Geometry-aware Transformer Decoder
        for i, decoder_block in enumerate(self.decoder):
            _, query_feature = decoder_block(
                torch.cat([query_coordinate, query_feature], dim=1), 
                torch.cat([key_coordinate, key_feature], dim=1)
            )
        
        return query_feature