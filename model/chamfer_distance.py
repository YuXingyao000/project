import torch
import torch.nn as nn
import torch.nn.functional as F


class ChamferDistanceL1(nn.Module):
    """
    Chamfer Distance Loss using L1 norm.
    Computes the bidirectional Chamfer distance between two point clouds.
    """
    
    def __init__(self, reduction='mean'):
        super(ChamferDistanceL1, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Compute Chamfer Distance between two point clouds.
        
        Args:
            pred: Predicted point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)
            
        Returns:
            Chamfer distance loss
        """
        # Ensure inputs are 3D tensors
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)  # (N, 3) -> (1, N, 3)
        if target.dim() == 2:
            target = target.unsqueeze(0)  # (M, 3) -> (1, M, 3)
        
        batch_size = pred.size(0)
        
        # Process large batches in chunks to avoid CUDA kernel limits
        chunk_size = 8  # Process 8 samples at a time
        total_loss = 0.0
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            pred_chunk = pred[i:end_idx]
            target_chunk = target[i:end_idx]
            
            # Compute pairwise distances between all points
            # pred_chunk: (chunk_size, N, 3), target_chunk: (chunk_size, M, 3)
            # dist: (chunk_size, N, M)
            dist = torch.cdist(pred_chunk, target_chunk, p=1)  # L1 norm
            
            # Forward direction: min distance from pred to target
            forward_dist = torch.min(dist, dim=2)[0]  # (chunk_size, N)
            
            # Backward direction: min distance from target to pred
            backward_dist = torch.min(dist, dim=1)[0]  # (chunk_size, M)
            
            # Sum both directions
            forward_loss = torch.sum(forward_dist, dim=1)  # (chunk_size,)
            backward_loss = torch.sum(backward_dist, dim=1)  # (chunk_size,)
            
            chunk_loss = forward_loss + backward_loss
            total_loss += torch.sum(chunk_loss)
        
        if self.reduction == 'mean':
            return total_loss / batch_size
        elif self.reduction == 'sum':
            return total_loss
        else:
            return total_loss


class ChamferDistanceL2(nn.Module):
    """
    Chamfer Distance Loss using L2 norm (Euclidean distance).
    Computes the bidirectional Chamfer distance between two point clouds.
    """
    
    def __init__(self, reduction='mean'):
        super(ChamferDistanceL2, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Compute Chamfer Distance between two point clouds.
        
        Args:
            pred: Predicted point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)
            
        Returns:
            Chamfer distance loss
        """
        # Ensure inputs are 3D tensors
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)  # (N, 3) -> (1, N, 3)
        if target.dim() == 2:
            target = target.unsqueeze(0)  # (M, 3) -> (1, M, 3)
        
        batch_size = pred.size(0)
        
        # Process large batches in chunks to avoid CUDA kernel limits
        chunk_size = 8  # Process 8 samples at a time
        total_loss = 0.0
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            pred_chunk = pred[i:end_idx]
            target_chunk = target[i:end_idx]
            
            # Compute pairwise distances between all points
            # pred_chunk: (chunk_size, N, 3), target_chunk: (chunk_size, M, 3)
            # dist: (chunk_size, N, M)
            dist = torch.cdist(pred_chunk, target_chunk, p=2)  # L2 norm
            
            # Forward direction: min distance from pred to target
            forward_dist = torch.min(dist, dim=2)[0]  # (chunk_size, N)
            
            # Backward direction: min distance from target to pred
            backward_dist = torch.min(dist, dim=1)[0]  # (chunk_size, M)
            
            # Sum both directions
            forward_loss = torch.sum(forward_dist, dim=1)  # (chunk_size,)
            backward_loss = torch.sum(backward_dist, dim=1)  # (chunk_size,)
            
            chunk_loss = forward_loss + backward_loss
            total_loss += torch.sum(chunk_loss)
        
        if self.reduction == 'mean':
            return total_loss / batch_size
        elif self.reduction == 'sum':
            return total_loss
        else:
            return total_loss


class ChamferDistanceWeighted(nn.Module):
    """
    Weighted Chamfer Distance Loss.
    Allows different weights for forward and backward directions.
    """
    
    def __init__(self, forward_weight=1.0, backward_weight=1.0, p=2, reduction='mean'):
        super(ChamferDistanceWeighted, self).__init__()
        self.forward_weight = forward_weight
        self.backward_weight = backward_weight
        self.p = p  # L1 (p=1) or L2 (p=2)
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Compute weighted Chamfer Distance between two point clouds.
        
        Args:
            pred: Predicted point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)
            
        Returns:
            Weighted Chamfer distance loss
        """
        # Ensure inputs are 3D tensors
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)  # (N, 3) -> (1, N, 3)
        if target.dim() == 2:
            target = target.unsqueeze(0)  # (M, 3) -> (1, M, 3)
        
        batch_size = pred.size(0)
        
        # Compute pairwise distances between all points
        dist = torch.cdist(pred, target, p=self.p)
        
        # Forward direction: min distance from pred to target
        forward_dist = torch.min(dist, dim=2)[0]  # (B, N)
        
        # Backward direction: min distance from target to pred
        backward_dist = torch.min(dist, dim=1)[0]  # (B, M)
        
        # Weighted sum
        forward_loss = self.forward_weight * torch.sum(forward_dist, dim=1)  # (B,)
        backward_loss = self.backward_weight * torch.sum(backward_dist, dim=1)  # (B,)
        
        total_loss = forward_loss + backward_loss
        
        if self.reduction == 'mean':
            return torch.mean(total_loss)
        elif self.reduction == 'sum':
            return torch.sum(total_loss)
        else:
            return total_loss


# Convenience function for backward compatibility
def chamfer_distance(pred, target, p=2, reduction='mean'):
    """
    Convenience function to compute Chamfer distance.
    
    Args:
        pred: Predicted point cloud (B, N, 3)
        target: Target point cloud (B, M, 3)
        p: Distance norm (1 for L1, 2 for L2)
        reduction: Reduction method ('mean', 'sum', or 'none')
        
    Returns:
        Chamfer distance
    """
    if p == 1:
        loss_fn = ChamferDistanceL1(reduction=reduction)
    else:
        loss_fn = ChamferDistanceL2(reduction=reduction)
    
    return loss_fn(pred, target) 