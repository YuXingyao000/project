"""
Query Generator module for point cloud completion.

This module implements the QueryGenerator component that generates query points
for completion by extracting global features and creating query embeddings.
"""

import torch
import torch.nn as nn


class QueryGenerator(nn.Module):
    """
    Query Generator component.
    
    This component generates query points for completion by:
    1. Extracting global features from encoded point features
    2. Generating coarse point cloud predictions
    3. Creating query embeddings from global features and coarse points
    """
    
    def __init__(self, embed_dim=384, num_query=224):
        """
        Initialize the Query Generator.
        
        Args:
            embed_dim (int): Embedding dimension for features
            num_query (int): Number of query points for completion
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_query = num_query
        
        # Feature dimension increase for global representation
        self.increase_dim = nn.Sequential(
            nn.Conv1d(embed_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        # Coarse point cloud prediction
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * num_query)
        )
        
        # Query feature processing MLP
        self.query_conv = nn.Sequential(
            nn.Conv1d(1024 + 3, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, embed_dim, 1)
        )

    def forward(self, encoded_features):
        """
        Forward pass of the Query Generator.
        
        Args:
            encoded_features (torch.Tensor): Encoded features from encoder [batch, embed_dim, num_points]
            
        Returns:
            tuple: (coarse_point_cloud, query_feature) where:
                - coarse_point_cloud: [batch, num_query, 3]
                - query_feature: [batch, embed_dim, num_query]
        """
        # Extract global feature
        global_feature = self.increase_dim(encoded_features)  # [batch, 1024, num_points//16]
        global_feature = torch.max(global_feature, dim=-1)[0]  # [batch, 1024]
        
        # Generate coarse point cloud
        coarse_point_cloud = self.coarse_pred(global_feature).reshape(-1, self.num_query, 3)
        
        # Create query features by concatenating global feature and coordinates
        query_feature = torch.cat([
            global_feature.unsqueeze(1).expand(-1, self.num_query, -1),
            coarse_point_cloud
        ], dim=-1)  # [batch, num_query, 3 + 1024]
        
        # Process query features through MLP
        query_feature = self.query_conv(query_feature.transpose(1, 2))  # [batch, embed_dim, num_query]
        
        return coarse_point_cloud, query_feature


if __name__ == "__main__":
    # Test the QueryGenerator component
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test input
    batch_size = 2
    embed_dim = 384
    num_points = 128  # num_points//16 from encoder
    test_input = torch.randn(batch_size, embed_dim, num_points).to(device)
    
    # Create QueryGenerator
    query_generator = QueryGenerator(
        embed_dim=embed_dim,
        num_query=224
    ).to(device)
    
    # Test forward pass
    with torch.no_grad():
        coarse_points, query_features = query_generator(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Coarse points shape: {coarse_points.shape}")
        print(f"Query features shape: {query_features.shape}")
        print(f"QueryGenerator parameters: {sum(p.numel() for p in query_generator.parameters()):,}") 