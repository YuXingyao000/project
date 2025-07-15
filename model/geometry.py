"""
Geometry utilities and position encoding for point cloud processing.

This module contains position encoding functions and other geometry-related
utilities used in the PCTransformer.
"""

import torch
import numpy as np


def sinusoidal_position_encoding(coordinates, encoding_dim=64):
    """
    Generate sinusoidal position encoding for point cloud coordinates.
    
    Reference: https://arxiv.org/pdf/2003.08934v2.pdf
    
    Args:
        coordinates (torch.Tensor): Input coordinates with shape [batch, 3, num_points]
        encoding_dim (int): Dimension of the position encoding
        
    Returns:
        torch.Tensor: Position encoding with shape [batch, 6*encoding_dim, num_points]
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
    """
    Extract coordinates and features from point cloud tensor.
    
    Args:
        points (torch.Tensor): Point cloud tensor with shape [batch, 3+feature_dim, num_points]
        
    Returns:
        tuple: (coordinates, features) where:
            - coordinates: [batch, 3, num_points]
            - features: [batch, feature_dim, num_points]
    """
    coordinates = points[:, :3, :]
    features = points[:, 3:, :]
    return coordinates, features


def combine_coordinates_and_features(coordinates, features):
    """
    Combine coordinates and features into a single point cloud tensor.
    
    Args:
        coordinates (torch.Tensor): Coordinates with shape [batch, 3, num_points]
        features (torch.Tensor): Features with shape [batch, feature_dim, num_points]
        
    Returns:
        torch.Tensor: Combined point cloud with shape [batch, 3+feature_dim, num_points]
    """
    return torch.cat([coordinates, features], dim=1) 