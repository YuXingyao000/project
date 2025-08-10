"""
Attention mechanisms for point cloud transformer.

This module contains the core attention components used in the PCTransformer,
including multi-head attention and various attention blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for point cloud processing.
    
    This module implements scaled dot-product attention with multiple heads,
    taking separate Q, K, V inputs for flexibility in self-attention and cross-attention.
    """
    def __init__(self, d_model, num_heads=8, attn_drop=0., proj_drop=0.):
        """
        Initialize the attention module.
        
        Args:
            d_model (int): Input and output feature dimension
            num_heads (int): Number of attention heads
            attn_drop (float): Dropout rate for attention weights
            proj_drop (float): Dropout rate for output projection
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
            query (torch.Tensor): Query tensor of shape [batch, heads, seq_len, head_dim]
            key (torch.Tensor): Key tensor of shape [batch, heads, seq_len, head_dim]
            value (torch.Tensor): Value tensor of shape [batch, heads, seq_len, head_dim]
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Attention output of shape [batch, heads, seq_len, head_dim]
        """
        # Compute attention scores: Q * K^T
        attention_scores = torch.einsum('b h i d, b h j d -> b h i j', query, key)
        
        # Scale the attention scores
        attention_scores = attention_scores * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
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
            q (torch.Tensor): Query tensor of shape [batch_size, seq_len, d_model]
            k (torch.Tensor): Key tensor of shape [batch_size, seq_len, d_model]
            v (torch.Tensor): Value tensor of shape [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model]
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


class FeedForward(nn.Module):
    """
    Feed-forward network with residual connection.
    
    Standard feed-forward network used in transformer blocks with
    two linear layers and a GELU activation function.
    """
    def __init__(self, in_channels, hidden_channels=None, dropout_rate=0.0):
        """
        Initialize the feed-forward network.
        
        Args:
            in_channels (int): Input feature dimension
            hidden_channels (int, optional): Hidden layer dimension. Defaults to 2 * in_channels
            dropout_rate (float): Dropout rate for regularization
        """
        super().__init__()
        hidden_channels = hidden_channels if hidden_channels is not None else in_channels * 2
        
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, in_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass through the feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, in_channels]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, in_channels]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x 