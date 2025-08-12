import torch
from pointnet2_ops import pointnet2_utils
import numpy as np

def knn_index(k_nearest_neighbors, source_coord, target_coord):
    """
    Args:
        - k_nearest_neighbors (int): Max sample number in local region
        - source_coord (torch.Tensor): All points [batch, 3, num_points]
        - target_coord (torch.Tensor): Query points [batch, 3, num_query]
    Returns:
        - group_idx (torch.Tensor): Grouped points index [batch, num_points, k_nearest_neighbors]
    """
    sqrdists = square_distance(source_coord, target_coord)
    _, group_idx = torch.topk(sqrdists, k_nearest_neighbors, dim=-1, largest=False, sorted=False)
    return group_idx


def square_distance(source, target):
    """
    Calculate Euclid distance between each two points.
    
    Args:
        - source (torch.Tensor): Source points [batch, num_points, 3]
        - target (torch.Tensor): Target points [batch, num_query, 3]
    Returns:
        - dist (torch.Tensor): Per-point square distance [batch, num_points, num_query]
    """
    B, _, SOURCE_NUM_POINTS = source.shape
    _, _, TARGET_NUM_POINTS = target.shape
    dist = -2 * torch.matmul(source.permute(0, 2, 1), target)
    dist += torch.sum(source ** 2, -2).view(B, SOURCE_NUM_POINTS, 1)
    dist += torch.sum(target ** 2, -2).view(B, 1, TARGET_NUM_POINTS)
    return dist  


def fps_downsample(points, sample_num):
    """
    Downsample the point cloud using FPS(Furthest Point Sampling).
    
    Args:
        - points (torch.Tensor): Point cloud [batch, 3 + feature_dim, num_points]
        - sample_num (int): Number of points to sample
    Returns:
        - new_points (torch.Tensor): Downsampled point cloud [batch, 3 + feature_dim, sample_num]
    """
    with torch.no_grad():
        coords = points[:, :3, :] # The first 3 channels are the coordinates
        xyz = coords.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, sample_num)
        points = points.contiguous()
        new_points = pointnet2_utils.gather_operation(points, fps_idx) # The so called gather_operation is just picking the features according to the fps_idx
        return new_points

def sinusoidal_position_encoding(coordinates, encoding_dim=64):
    """
    Generate sinusoidal position encoding for point cloud coordinates.
    
    Reference: https://arxiv.org/pdf/2003.08934v2.pdf
    """
    # Normalize coordinates to [-1, 1] range, batch-wise
    coor_min = coordinates.min(dim=-1, keepdim=True)[0]
    coor_max = coordinates.max(dim=-1, keepdim=True)[0]
    normalized_coor = 2 * ((coordinates - coor_min) / (coor_max - coor_min)) - 1

    # Define sinusoidal wave frequencies
    freqs = torch.arange(encoding_dim, dtype=torch.float, device=coordinates.device)
    freqs = np.pi * (2 ** freqs)

    # Reshape frequencies for broadcasting
    freqs = freqs.view(*[1] * len(normalized_coor.shape), -1)  # 1 x 1 x 1 x encoding_dim
    normalized_coor = normalized_coor.unsqueeze(-1)  # batch x 3 x num_points x 1
    
    # Compute sinusoidal values
    k = normalized_coor * freqs  # batch x 3 x num_points x encoding_dim
    sin_values = torch.sin(k)  # batch x 3 x num_points x encoding_dim
    cos_values = torch.cos(k)  # batch x 3 x num_points x encoding_dim
    
    # Concatenate sin and cos values
    combined = torch.cat([sin_values, cos_values], dim=-1)  # batch x 3 x num_points x (2*encoding_dim)
    
    # Reshape to final format
    position_encoding = combined.transpose(-1, -2).reshape(
        coordinates.shape[0], -1, coordinates.shape[-1]
    )  # batch x (6*encoding_dim) x num_points
    
    return position_encoding


def extract_coordinates_and_features(points):
    coordinates = points[:, :3, :]
    features = points[:, 3:, :]
    return coordinates, features


def combine_coordinates_and_features(coordinates, features):
    return torch.cat([coordinates, features], dim=1) 

def jitter_points(pc, std=0.01, clip=0.05):
    """
    Jitter the points in the point cloud.
    
    Args:
        - pc (torch.Tensor): Point cloud [batch, num_points, 3]
        - std (float): Standard deviation of the noise
        - clip (float): Clipping value
    Returns:
        - pc (torch.Tensor): Jittered point cloud [batch, num_points, 3]
    """
    jittered_data = torch.normal(mean=0.0, std=std, size=pc.shape, 
                               dtype=pc.dtype, device=pc.device).clamp(-clip, clip)
    pc[:, :, :3] += jittered_data
    return pc
