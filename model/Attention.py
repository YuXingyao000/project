"""
Attention mechanisms for point cloud transformer.

This module contains the core attention components used in the PCTransformer,
including multi-head attention and various attention blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.DGCNN import kNNQuery

class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention module.
    """
    def __init__(self, d_model, num_heads=8, attn_drop=0., proj_drop=0.):
        """
        Args:
            - d_model (int): Input and output feature dimension
            - num_heads (int): Number of attention heads
            - attn_drop (float): Dropout rate for attention weights
            - proj_drop (float): Dropout rate for output projection
        """
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.scale = self.d_k ** -0.5

        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(attn_drop)
        self.output_dropout = nn.Dropout(proj_drop)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """
        Compute attention weights and apply them to values.
        
        Args:
            - query (torch.Tensor): Query tensor of shape [batch, heads, seq_len, head_dim]
            - key (torch.Tensor): Key tensor of shape [batch, heads, seq_len, head_dim]
            - value (torch.Tensor): Value tensor of shape [batch, heads, seq_len, head_dim]
            - mask (torch.Tensor, optional): Attention mask
            
        Returns:
            - attention_output (torch.Tensor): Attention output of shape [batch, heads, seq_len, head_dim]
        """
        # Compute attention scores: Q * K^T
        attention_scores = torch.einsum('b h i d, b h j d -> b h i j', query, key)
        
        # Scale the attention scores
        attention_scores = attention_scores * self.scale
        
        # Apply mask if provided
        if mask is not None:
            # 1 for mask, 0 for not mask
            # mask shape N, N
            mask_value = -torch.finfo(attention_scores.dtype).max
            mask = (mask > 0)  # convert to boolen, shape torch.BoolTensor[N, N]
            attention_scores = attention_scores.masked_fill(mask, mask_value) # B h N N
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, value)
        
        return attention_output

    def forward(self, q, k, v, mask=None):
        """
        Forward pass of the attention mechanism.
        
        Args:
            - q (torch.Tensor): Query tensor of shape [batch_size, seq_len, d_model]
            - k (torch.Tensor): Key tensor of shape [batch_size, seq_len, d_model]
            - v (torch.Tensor): Value tensor of shape [batch_size, seq_len, d_model]
            - mask (torch.Tensor, optional): Attention mask
            
        Returns:
            - output (torch.Tensor): Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Step 1: Reshape Q, K, V for multi-head attention
        # Rearrange: [batch, seq, d_model] -> [batch, heads, seq, d_k]
        q = rearrange(q, 'batch seq (heads d_k) -> batch heads seq d_k', 
                     heads=self.num_heads, d_k=self.d_k)
        k = rearrange(k, 'batch seq (heads d_k) -> batch heads seq d_k', 
                     heads=self.num_heads, d_k=self.d_k)
        v = rearrange(v, 'batch seq (heads d_k) -> batch heads seq d_k', 
                     heads=self.num_heads, d_k=self.d_k)
        
        # Step 2: Compute attention
        attention_output = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Step 3: Reshape back to original format
        # Rearrange: [batch, heads, seq, d_k] -> [batch, seq, d_model]
        output = rearrange(attention_output, 'batch heads seq d_k -> batch seq (heads d_k)')
        
        # Step 4: Apply output projection and dropout
        output = self.output_projection(output)
        output = self.output_dropout(output)
        
        return output

class GraphAttention(nn.Module):
    """
    Graph Attention Module.
    
    This is the key where PoinTr integrates geometry information.
    """
    def __init__(self, d_model, k_nearest_neighbors=8):
        """
        Args:
            - d_model (int): Input and output feature dimension
            - k_nearest_neighbors (int): Number of nearest neighbors to consider
        """
        super().__init__()
        self.kNN_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.kNNQuery = kNNQuery(k_nearest_neighbors=k_nearest_neighbors)
    
    def forward(self, query_coords, query_features, key_coords, key_features, denoise_length=None):
        """
        Args:
            - query_coords (torch.Tensor): Query point coordinates [batch, 3, num_query]
            - query_features (torch.Tensor): Query point features [batch, feature_dim, num_query]
            - key_coords (torch.Tensor): Key point coordinates [batch, 3, num_points]
            - key_features (torch.Tensor): Key point features [batch, feature_dim, num_points]
            - denoise_length (int, optional): Length of the denoised query points. Defaults to None. This is used for noising auxiliary task.
        
        Returns:
            - geom_features (torch.Tensor): Geometry features [batch, num_points, feature_dim]
        """
        geom_features = self.kNNQuery(query_coords, query_features, key_coords, key_features, denoise_length=denoise_length)
        geom_features = rearrange(geom_features, 'batch double_feature_dim num_points k -> batch k num_points double_feature_dim')
        geom_features = self.kNN_proj(geom_features)  # [batch, k, num_points, feature_dim]
        geom_features = geom_features.max(dim=1, keepdim=False)[0]  # [batch, num_points, feature_dim]
        return geom_features

class FeedForward(nn.Module):
    """
    Standard feed-forward network with residual connection.
    """
    def __init__(self, in_channels, hidden_channels=None, dropout_rate=0.0):
        """
        Args:
            - in_channels (int): Input feature dimension
            - hidden_channels (int, optional): Hidden layer dimension. Defaults to 2 * in_channels
            - dropout_rate (float): Dropout rate for regularization
        """
        super().__init__()
        hidden_channels = hidden_channels if hidden_channels is not None else in_channels * 2
        
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, in_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Args:
            - x (torch.Tensor): Input tensor of shape [batch_size, seq_len, in_channels]
            
        Returns:
            - output (torch.Tensor): Output tensor of shape [batch_size, seq_len, in_channels]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x 