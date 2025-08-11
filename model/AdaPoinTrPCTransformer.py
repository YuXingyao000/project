import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from model.GAEncoder import GeometryAwareTransformerEncoder
from model.GADecoder import GeometryAwareTransformerDecoder
from model.QueryGenerator import DynamicQueryGenerator



class AdaPoinTrPCTransformer(nn.Module):
    """
    AdaPoinTrPCTransformer: Composed Point Cloud Transformer for completion tasks.
    
    This model combines three main components:
    1. GeometryAwareTransformerEncoder: Processes incomplete point cloud
    2. QueryGenerator: Generates query points for completion
    3. GeometryAwareTransformerDecoder: Refines query points
    """
    
    def __init__(self, in_chans=3, embed_dim=384, depth=[[1, 5], [1, 7]], num_heads=6, grouper_k_nearest_neighbors=16, downsample_divisor=[4, 8], num_query=224):
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
            num_heads=num_heads,
            downsample_divisor=downsample_divisor,
            grouper_k_nearest_neighbors=grouper_k_nearest_neighbors
        )
        
        self.query_generator = DynamicQueryGenerator(
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
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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