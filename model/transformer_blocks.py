"""
Transformer blocks for point cloud processing.

This module contains various transformer block implementations including
self-attention, cross-attention, and geometry-aware attention blocks.
"""

import torch
import torch.nn as nn
from einops import rearrange

from model.attention import MultiHeadAttention, FeedForward
from model.DGCNN import kNNQuery


class SelfAttentionBlock(nn.Module):
    """
    Standard self-attention transformer block.
    
    Implements a vanilla transformer block with self-attention and
    feed-forward network with residual connections.
    """
    def __init__(self, d_model=384, num_heads=6):
        """
        Initialize the self-attention block.
        
        Args:
            d_model (int): Feature dimension
            num_heads (int): Number of attention heads
        """
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads
        )
        self.feed_forward = FeedForward(d_model, hidden_channels=d_model * 2)
        self.feed_forward_norm = nn.LayerNorm(d_model)
    
    def forward(self, points):
        """
        Forward pass for vanilla transformer block.
        
        Args:
            points (torch.Tensor): Input points with shape [batch, 3+feature_dim, num_points]
            
        Returns:
            torch.Tensor: Output points with shape [batch, 3+feature_dim, num_points]
        """
        # Extract coordinates and features
        coords = points[:, :3, :]  # [batch, 3, num_points]
        features = points[:, 3:, :]  # [batch, feature_dim, num_points]
        
        # Normalize features
        norm_features = self.input_norm(features.transpose(1, 2))  # [batch, num_points, feature_dim]
        
        # Self-attention
        qkv = self.qkv_proj(norm_features)  # [batch, num_points, 3*feature_dim]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch, num_points, feature_dim]
        attn_features = self.multi_head_attention(q, k, v)  # [batch, num_points, feature_dim]
        
        # Residual connection
        output_features = features.transpose(1, 2) + attn_features  # [batch, num_points, feature_dim]
        
        # Feed-forward network
        output_features = self.feed_forward(self.feed_forward_norm(output_features)) + output_features
        
        return torch.cat([coords, output_features.transpose(1, 2)], dim=1)  # [batch, 3+feature_dim, num_points]


class GeometryAwareSelfAttentionBlock(nn.Module):
    """
    Geometry-aware self-attention transformer block.
    
    Extends the standard self-attention block with geometry-aware features
    computed using k-nearest neighbors.
    """
    def __init__(self, d_model=384, num_heads=6, k_neighbors=8):
        """
        Initialize the geometry-aware self-attention block.
        
        Args:
            d_model (int): Feature dimension
            num_heads (int): Number of attention heads
            k_neighbors (int): Number of k-nearest neighbors for geometry features
        """
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads
        )
        self.feed_forward = FeedForward(d_model, hidden_channels=d_model * 2)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        
        # Geometry-aware components
        self.kNNQuery = kNNQuery(k_nearest_neighbors=k_neighbors)
        self.kNN_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.merge_proj = nn.Linear(d_model * 2, d_model)
        
    def forward(self, points):
        """
        Forward pass for geometry-aware transformer block.
        
        Args:
            points (torch.Tensor): Input points with shape [batch, 3+feature_dim, num_points]
            
        Returns:
            torch.Tensor: Output points with shape [batch, 3+feature_dim, num_points]
        """
        # Extract coordinates and features
        coords = points[:, :3, :]  # [batch, 3, num_points]
        features = points[:, 3:, :]  # [batch, feature_dim, num_points]
        
        # Normalize features
        norm_features = self.input_norm(features.transpose(1, 2))  # [batch, num_points, feature_dim]
        
        # Self-attention
        qkv = self.qkv_proj(norm_features)  # [batch, num_points, 3*feature_dim]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch, num_points, feature_dim]
        attn_features = self.multi_head_attention(q, k, v)  # [batch, num_points, feature_dim]
        
        # Geometry-aware features using kNN
        # Prepare input for kNN: concatenate coords and normalized features
        normed_points = torch.cat([coords, norm_features.transpose(1, 2)], dim=1)  # [batch, 3+feature_dim, num_points]
        
        # Get geometry features
        geom_features = self.kNNQuery(normed_points, normed_points)  # [batch, 2*feature_dim, num_points, k]
        geom_features = rearrange(geom_features, 'batch double_feature_dim num_points k -> batch k num_points double_feature_dim')
        geom_features = self.kNN_proj(geom_features)  # [batch, k, num_points, feature_dim]
        geom_features = geom_features.max(dim=1, keepdim=False)[0]  # [batch, num_points, feature_dim]
        
        # Merge attention and geometry features
        merged_features = torch.cat([attn_features, geom_features], dim=-1)  # [batch, num_points, 2*feature_dim]
        merged_features = self.merge_proj(merged_features)  # [batch, num_points, feature_dim]
        
        # Residual connection
        output_features = merged_features + attn_features  # [batch, num_points, feature_dim]
        
        # Feed-forward network
        output_features = self.feed_forward_norm(output_features)
        output_features = self.feed_forward(output_features) + output_features
        
        # Prepare output: concatenate coords and features
        output = torch.cat([coords, output_features.transpose(1, 2)], dim=1)  # [batch, 3+feature_dim, num_points]
        
        return output


class CrossAttentionBlock(nn.Module):
    """
    Standard cross-attention transformer block.
    
    Implements cross-attention between query and key points with
    separate normalization and projection layers.
    """
    def __init__(self, d_model=384, num_heads=6, attn_drop=0., proj_drop=0.):
        """
        Initialize the cross-attention block.
        
        Args:
            d_model (int): Feature dimension
            num_heads (int): Number of attention heads
            attn_drop (float): Attention dropout rate
            proj_drop (float): Projection dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.query_norm = nn.LayerNorm(d_model)
        self.key_norm = nn.LayerNorm(d_model)
        
        self.feed_forward = FeedForward(d_model, hidden_channels=d_model * 2)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        
        self.q_map = nn.Linear(d_model, d_model, bias=False)
        self.k_map = nn.Linear(d_model, d_model, bias=False)
        self.v_map = nn.Linear(d_model, d_model, bias=False)
        
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads
        )
        
    def forward(self, query_points, key_points):
        """
        Forward pass for cross-attention block.
        
        Args:
            query_points (torch.Tensor): Query points with shape [batch, 3+feature_dim, num_query]
            key_points (torch.Tensor): Key points with shape [batch, 3+feature_dim, num_key]
            
        Returns:
            torch.Tensor: Updated query points with shape [batch, 3+feature_dim, num_query]
        """
        query_coords = query_points[:, :3, :]
        key_coords = key_points[:, :3, :]
        query_features = query_points[:, 3:, :]
        key_features = key_points[:, 3:, :]
        
        query_features = self.query_norm(query_features.transpose(1,2))
        key_features = self.key_norm(key_features.transpose(1,2))
        
        q = self.q_map(query_features)
        k = self.k_map(key_features)
        v = self.v_map(key_features)
        attn_features = self.multi_head_attention(q, k, v)
         
        query_features = query_features + attn_features
        query_features = query_features + self.feed_forward(self.feed_forward_norm(query_features))
        
        return torch.cat([query_coords, query_features.transpose(1,2)], dim=1)


class GeometryAwareCrossAttentionBlock(nn.Module):
    """
    Geometry-aware cross-attention transformer block.
    
    Combines self-attention and cross-attention with geometry-aware features
    computed using k-nearest neighbors for both self and cross attention.
    """
    def __init__(self, d_model=384, num_heads=6, k_neighbors=8):
        """
        Initialize the geometry-aware cross-attention block.
        
        Args:
            d_model (int): Feature dimension
            num_heads (int): Number of attention heads
            k_neighbors (int): Number of k-nearest neighbors for geometry features
        """
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads
        )
        self.feed_forward = FeedForward(d_model, hidden_channels=d_model * 2)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        
        # Geometry-aware components
        self.self_kNNQuery = kNNQuery(k_nearest_neighbors=k_neighbors)
        self.cross_kNNQuery = kNNQuery(k_nearest_neighbors=k_neighbors)
        self.kNN_proj1 = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.kNN_proj2 = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.self_merge_proj = nn.Linear(d_model * 2, d_model)
        self.cross_merge_proj = nn.Linear(d_model * 2, d_model)
        
        # Cross-attention components
        self.cross_norm_q = nn.LayerNorm(d_model)
        self.cross_norm_k = nn.LayerNorm(d_model)
        self.cross_q_map = nn.Linear(d_model, d_model, bias=False)
        self.cross_k_map = nn.Linear(d_model, d_model, bias=False)
        self.cross_v_map = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, query_points, key_points):
        """
        Forward pass for geometry-aware cross-attention block.
        
        Args:
            query_points (torch.Tensor): Query points with shape [batch, 3+feature_dim, num_query]
            key_points (torch.Tensor): Key points with shape [batch, 3+feature_dim, num_key]
            
        Returns:
            torch.Tensor: Updated query points with shape [batch, 3+feature_dim, num_query]
        """
        #################################
        # Geometry-aware Self-attention #
        #################################
        # Extract coordinates and features
        query_coords = query_points[:, :3, :]  # [batch, 3, num_points]
        key_coords = key_points[:, :3, :]  # [batch, 3, num_points]
        query_features = query_points[:, 3:, :]  # [batch, feature_dim, num_points]
        key_features = key_points[:, 3:, :]  # [batch, feature_dim, num_points]
        
        # Normalize features
        norm_features = self.input_norm(query_features.transpose(1, 2))  # [batch, num_points, feature_dim]
        
        # Self-attention
        qkv = self.qkv_proj(norm_features)  # [batch, num_points, 3*feature_dim]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch, num_points, feature_dim]
        attn_features = self.multi_head_attention(q, k, v)  # [batch, num_points, feature_dim]
        
        # Geometry-aware features using kNN
        # Prepare input for kNN: concatenate coords and normalized features
        normed_points = torch.cat([query_coords, norm_features.transpose(1, 2)], dim=1)  # [batch, 3+feature_dim, num_points]
        
        # Get geometry features
        geom_features = self.self_kNNQuery(normed_points, normed_points)  # [batch, 2*feature_dim, num_points, k]
        geom_features = rearrange(geom_features, 'batch double_feature_dim num_points k -> batch k num_points double_feature_dim')
        geom_features = self.kNN_proj1(geom_features)  # [batch, k, num_points, feature_dim]
        geom_features = geom_features.max(dim=1, keepdim=False)[0]  # [batch, num_points, feature_dim]
        
        # Merge attention and geometry features
        attn_features = torch.cat([attn_features, geom_features], dim=-1)  # [batch, num_points, 2*feature_dim]
        attn_features = self.self_merge_proj(attn_features)  # [batch, num_points, feature_dim]
        
        # Residual connection
        query_features = attn_features + query_features.transpose(1,2)  # [batch, num_points, feature_dim]
        
        #################################
        # Geometry-aware Cross-attention #
        #################################
        
        normed_query_features = self.cross_norm_q(query_features)
        normed_key_features = self.cross_norm_k(key_features.transpose(1,2))
        
        cross_q = self.cross_q_map(normed_query_features)
        cross_k = self.cross_k_map(normed_key_features)
        cross_v = self.cross_v_map(normed_key_features)
        cross_features = self.multi_head_attention(cross_q, cross_k, cross_v)
        
        cross_query_points = torch.cat([query_coords, normed_query_features.transpose(1,2)], dim=1)
        cross_key_points = torch.cat([key_coords, normed_key_features.transpose(1,2)], dim=1)
        cross_geom_features = self.cross_kNNQuery(cross_query_points, cross_key_points)
        cross_geom_features = rearrange(cross_geom_features, 'batch double_feature_dim num_points k -> batch k num_points double_feature_dim')
        cross_geom_features = self.kNN_proj2(cross_geom_features)  # [batch, k, num_points, feature_dim]
        cross_geom_features = cross_geom_features.max(dim=1, keepdim=False)[0]  # [batch, num_points, feature_dim]
        
        cross_features = torch.cat([cross_features, cross_geom_features], dim=-1)  # [batch, num_points, 2*feature_dim]
        cross_features = self.cross_merge_proj(cross_features)  # [batch, num_points, feature_dim]
        
        query_features = query_features + cross_features
        query_features = query_features + self.feed_forward(self.feed_forward_norm(query_features))
        
        return torch.cat([query_coords, query_features.transpose(1,2)], dim=1) 