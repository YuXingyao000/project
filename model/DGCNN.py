import torch
from torch import nn
from model.Utils import fps_downsample, knn_index
from einops import rearrange, repeat

from model.Utils import extract_coordinates_and_features

class kNNQuery(nn.Module):
    """
    K-Nearest Neighbor Query module for geometry-aware transformer.
    
    This module finds k-nearest neighbors and computes edge features
    by gathering neighbor information and computing local-global feature differences.
    """
    def __init__(self, k_nearest_neighbors=8):
        """
        Args:
            - k_nearest_neighbors (int): Number of nearest neighbors to query
        """
        super().__init__()
        self.k_nearest_neighbors = k_nearest_neighbors
    
    def _find_knn_indices(self, query_coords, key_coords):
        """
        Find k-nearest neighbor indices between query and key points.
        
        Args:
            - query_coords (torch.Tensor): Query point coordinates [batch, 3, num_query]
            - key_coords (torch.Tensor): Key point coordinates [batch, 3, num_key]
            
        Returns:
            - flattened_indices (torch.Tensor): Flattened indices for gathering neighbor features
        """
        with torch.no_grad():
            # Find k-nearest neighbors: query -> key
            knn_indices = knn_index(self.k_nearest_neighbors, query_coords, key_coords)  # [batch, num_query, k]
            knn_indices = knn_indices.transpose(-1, -2).contiguous()  # [batch, k, num_query]
            
            # Create batch offset indices for flattening
            batch_size = query_coords.shape[0]
            num_key_points = key_coords.shape[2]
            batch_offsets = torch.arange(0, batch_size, device=query_coords.device).view(-1, 1, 1) * num_key_points
            
            # Flatten indices for efficient gathering
            flattened_indices = knn_indices + batch_offsets
            flattened_indices = flattened_indices.view(-1)
            
        return flattened_indices
    
    def _gather_neighbor_features(self, key_features, num_query_points, knn_indices):
        """
        Gather neighbor features using k-nearest neighbor indices.
        
        Args:
            - key_features (torch.Tensor): Key point features [batch, feature_dim, num_key]
            - query_features (torch.Tensor): Query point features [batch, feature_dim, num_query]
            - knn_indices (torch.Tensor): Flattened k-nearest neighbor indices
            
        Returns:
            - neighbor_features (torch.Tensor): Neighbor features [batch, feature_dim, num_query, k]
        """
        batch_size, feature_dim, num_key_points = key_features.shape
        
        # Reshape key features for indexing using einops
        key_features_flat = rearrange(key_features, 'batch feature_dim num_key -> (batch num_key) feature_dim')
        
        # Gather neighbor features using indices
        neighbor_features = key_features_flat[knn_indices, :]  # [(batch * k * num_query), feature_dim]
        
        # Reshape to [batch, k, num_query, feature_dim] then permute
        neighbor_features = rearrange(
            neighbor_features, 
            '(batch k num_query) feature_dim -> batch feature_dim num_query k',
            batch=batch_size, 
            k=self.k_nearest_neighbors, 
            num_query=num_query_points
        )
        
        return neighbor_features
    
    def _compute_edge_features(self, neighbor_features, query_features):
        """
        Compute edge features by concatenating local and global information.
        
        Args:
            - neighbor_features (torch.Tensor): Neighbor features [batch, feature_dim, num_query, k]
            - query_features (torch.Tensor): Query point features [batch, feature_dim, num_query]
            
        Returns:
            - edge_features (torch.Tensor): Edge features [batch, 2*feature_dim, num_query, k]
        """
        # Expand query features to match neighbor features shape
        query_features_expanded = repeat(
            query_features, 
            'batch feature_dim num_query -> batch feature_dim num_query k', 
            k=self.k_nearest_neighbors
        )
        
        # Compute edge features: concatenate [neighbor - query, query]
        edge_features = torch.cat([
            neighbor_features - query_features_expanded,  # Local geometric information
            query_features_expanded                       # Global context
        ], dim=1)  # [batch, 2*feature_dim, num_query, k]
        
        return edge_features
    
    def forward(self, query_coords, query_features, key_coords, key_features, denoise_length=None):
        """
        Forward pass of kNN Query.
        
        Args:
            - query_coords (torch.Tensor): Query point coordinates [batch, 3, num_query]
            - query_features (torch.Tensor): Query point features [batch, feature_dim, num_query]
            - key_coords (torch.Tensor): Key point coordinates [batch, 3, num_key]
            - key_features (torch.Tensor): Key point features [batch, feature_dim, num_key]
            - denoise_length (int, optional): Length of the denoised query points. Defaults to None. This is used for noising auxiliary task.
                               
        Returns:
            - edge_features (torch.Tensor): Edge features with shape [batch, 2*feature_dim, num_query, k]
        """
        assert len(query_coords.shape) == 3 and query_coords.shape[1] == 3, f"Query coordinates must have shape [batch, 3, num_query], got {query_coords.shape}"
        assert len(key_coords.shape) == 3 and key_coords.shape[1] == 3, f"Key coordinates must have shape [batch, 3, num_key], got {key_coords.shape}"
        
        if denoise_length is None:
            # Find k-nearest neighbors
            knn_indices = self._find_knn_indices(query_coords, key_coords)

            # Gather neighbor features
            num_query_points = query_coords.shape[2]
            neighbor_features = self._gather_neighbor_features(key_features, num_query_points, knn_indices)

            # Compute edge features
            edge_features = self._compute_edge_features(neighbor_features, query_features)
        else:
            knn_indices_raw = self._find_knn_indices(query_coords[:, :, :-denoise_length], key_coords[:, :, :-denoise_length])
            knn_indices_noised = self._find_knn_indices(query_coords[:, :, -denoise_length:], key_coords)
            
            num_query_points_raw = query_coords.shape[2] - denoise_length
            neighbor_features_raw = self._gather_neighbor_features(key_features[:, :, :-denoise_length], num_query_points_raw, knn_indices_raw)
            neighbor_features_noised = self._gather_neighbor_features(key_features, denoise_length, knn_indices_noised)
            
            mix_neighbor_features = torch.cat([neighbor_features_raw, neighbor_features_noised], dim=-2)
            edge_features = self._compute_edge_features(mix_neighbor_features, query_features)
        
        return edge_features


class EdgeConv(nn.Module):
    """
    Edge Convolution module for point cloud processing.
    
    This module computes edge features by finding k-nearest neighbors and
    aggregating local geometric information around each point through
    convolution followed by max pooling.
    
    Ref: https://arxiv.org/abs/1801.07829
    """
    def __init__(self, in_channels, out_channels, k_nearest_neighbors=16):
        """
        Args:
            - in_channels (int): Number of input feature channels
            - out_channels (int): Number of output feature channels
            - k_nearest_neighbors (int): Number of nearest neighbors to consider
        """
        super().__init__()
        self.k_nearest_neighbors = k_nearest_neighbors
        
        # Edge feature processing layers
        self.kNNQuery = kNNQuery(k_nearest_neighbors)
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False)
        self.normalization = nn.GroupNorm(4, out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
    
    def forward(self, query_coords, query_features, key_coords, key_features):
        """
        Forward pass of Edge Convolution.
        
        Args:
            - query_coords (torch.Tensor): Query point coordinates [batch, 3, num_query]
            - query_features (torch.Tensor): Query point features [batch, feature_dim, num_query]
            - key_coords (torch.Tensor): Key point coordinates [batch, 3, num_key]
            - key_features (torch.Tensor): Key point features [batch, feature_dim, num_key]
                               
        Returns:
            - query_coords (torch.Tensor): Query point coordinates [batch, 3, num_query]
            - edge_features (torch.Tensor): Edge features [batch, out_channels, num_query, k]
        """
        edge_features = self.kNNQuery(query_coords, query_features, key_coords, key_features)
        
        # Process edge features through convolution layers
        edge_features = self.conv(edge_features)  # [batch, out_channels, num_query, k]
        edge_features = self.normalization(edge_features)
        edge_features = self.activation(edge_features)
        
        # Aggregate edge features across neighbors using max pooling
        edge_features = edge_features.max(dim=-1, keepdim=False)[0]  # [batch, out_channels, num_query]
        
        return query_coords, edge_features
            


class DGCNN_Grouper(nn.Module):
    """
    This is a light-weight DGCNNgrouper that groups points into a proxy.
    
    Args:
        - input_dim (int): Number of input feature channels
        - output_dim (int): Number of output feature channels
        - grouper_downsample (list): Downsample divisor for the grouper
        - k_nearest_neighbors (int): Number of nearest neighbors to consider
    """
    def __init__(self, input_dim=3, output_dim=128, grouper_downsample=[4, 16], k_nearest_neighbors=16):
        super().__init__()
        self.input_trans = nn.Conv1d(input_dim, 8, 1)
        self.downsample_divisor1 = grouper_downsample[0]
        self.downsample_divisor2 = grouper_downsample[1]
        self.edge_conv1 = EdgeConv(in_channels=8, out_channels=32, k_nearest_neighbors=k_nearest_neighbors)
        self.edge_conv2 = EdgeConv(in_channels=32, out_channels=64, k_nearest_neighbors=k_nearest_neighbors)
        self.edge_conv3 = EdgeConv(in_channels=64, out_channels=64, k_nearest_neighbors=k_nearest_neighbors)
        self.edge_conv4 = EdgeConv(in_channels=64, out_channels=output_dim, k_nearest_neighbors=k_nearest_neighbors)
        
    def forward(self, point_cloud):
        """
        Forward pass of DGCNN_Grouper.
        
        Args:
            - point_cloud (torch.Tensor): Point cloud [batch, 3, num_points]
        
        Returns:
            - query_coords (torch.Tensor): Query point coordinates [batch, 3, num_points // downsample_divisor2]
            - query_features (torch.Tensor): Query point features [batch, output_dim, num_points // downsample_divisor2]
        """
        assert len(point_cloud.shape) == 3 and point_cloud.shape[1] == 3, f"Point cloud must have shape [batch, 3, num_points], got {point_cloud.shape}"
        
        num_points = point_cloud.shape[2]
        coords = point_cloud # bs 3 num_points
        feature = self.input_trans(point_cloud) # bs 8 num_points
        
        key_coords, key_features = self.edge_conv1(coords, feature, coords, feature) # bs, [3 + 32], num_points
        query_points = fps_downsample(torch.cat([key_coords, key_features], dim=1), num_points // self.downsample_divisor1) # bs, [3 + 32], num_points // 4
        query_coords, query_features = extract_coordinates_and_features(query_points)
        query_coords, query_features = self.edge_conv2(query_coords, query_features, key_coords, key_features) # bs, [3 + 64], num_points // 4
        
        key_coords, key_features = self.edge_conv3(query_coords, query_features, query_coords, query_features) # bs, [3 + 64], num_points // 16
        query_points = fps_downsample(torch.cat([key_coords, key_features], dim=1), num_points // self.downsample_divisor2) # bs, [3 + 64], num_points // 16
        query_coords, query_features = extract_coordinates_and_features(query_points)
        query_coords, query_features = self.edge_conv4(query_coords, query_features, query_coords, query_features) # bs, [3 + 128], num_points // 16
        
        
        return query_coords, query_features
            

if __name__ == "__main__":
    x = torch.randn(64, 3, 2048)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    model = DGCNN_Grouper().to(device)
    points = model(x)
    print(points.shape) # bs, [3 + 128], num_points // 16