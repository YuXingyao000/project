import torch.nn as nn

from model.TransformerBlocks import (
    CrossAttentionBlock,
    GeometryAwareCrossAttentionBlock
)

class GeometryAwareTransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, depth=[1, 7], attention_k_nearest_neighbors=8, norm_eps=1e-5):
        """
        Args:
            - embed_dim (int): Embedding dimension for features
            - num_heads (int): Number of attention heads
            - depth (list): List of [geom_blocks, vanilla_blocks]
            - attention_k_nearest_neighbors (int): Number of nearest neighbors to use for attention blocks
            - norm_eps (float): Epsilon for layer normalization
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.geom_depth = depth[0]
        # Transformer decoder blocks
        self.decoder = nn.ModuleList(
            [GeometryAwareCrossAttentionBlock(d_model=embed_dim, num_heads=num_heads, k_nearest_neighbors=attention_k_nearest_neighbors, norm_eps=norm_eps) for _ in range(depth[0])] +
            [CrossAttentionBlock(d_model=embed_dim, num_heads=num_heads, norm_eps=norm_eps) for _ in range(depth[1])]
        )

    def forward(self, query_coordinate, query_feature, key_coordinate, key_feature, mask=None, denoise_length=None):
        """
        Args:
            - query_coordinate (torch.Tensor): Query coordinates [batch, num_query, 3]
            - query_feature (torch.Tensor): Query features [batch, num_query, embed_dim]
            - key_coordinate (torch.Tensor): Key coordinates [batch, num_points, 3]
            - key_feature (torch.Tensor): Key features [batch, num_points, embed_dim]
            
        Returns:
            - query_feature (torch.Tensor): Query features [batch, embed_dim, num_query]
        """
        # Geometry-aware Transformer Decoder
        for i, decoder_block in enumerate(self.decoder):
            if i < self.geom_depth:
                query_feature = decoder_block(
                    query_coordinate, query_feature, key_coordinate, key_feature, mask=mask, denoise_length=denoise_length
                )
            else:
                query_feature = decoder_block(
                    query_feature, key_feature, mask=mask
                )
        
        return query_feature