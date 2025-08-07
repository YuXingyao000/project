import numpy as np
import torch
import torch.nn.functional as F
import random


class Partial:
    """Transformations for point cloud data during training."""
    
    def __init__(self, target_points: int = 2048, min_ratio: float = 0.25, max_ratio: float = 0.75, padding_zeros: bool = False):
        """
        Initialize the point cloud transform.
        Generate partial point cloud using random center points.
        Args:
            target_points: Number of points to downsample to (default: 2048)
            min_ratio: Minimum ratio of points to sample (default: 0.25)
            max_ratio: Maximum ratio of points to sample (default: 0.75)
            padding_zeros: Whether to pad with zeros instead of removing points (default: False)
        """
        self.target_points = target_points
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.padding_zeros = padding_zeros

    def __call__(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Apply the transformation to a point cloud.
        
        Args:
            point_cloud: Input point cloud of shape (N, 3) where N >= target_points
            
        Returns:
            Transformed point cloud of shape (target_points, 3)
        """
        # Convert to torch tensor and add batch dimension
        xyz = torch.from_numpy(point_cloud).float().unsqueeze(0)  # (1, N, 3)
        
        # Get the number of points in the input
        num_points = xyz.shape[1]
        assert num_points == 8192, f"Point cloud must have 8192 points, but has {num_points} points"
        
        # Calculate crop range
        complete_points = 8192
        min_crop = int(complete_points * self.min_ratio)  # 2048
        max_crop = int(complete_points * self.max_ratio)  # 6144
        
        # Randomly choose number of points to crop
        num_crop = random.randint(min_crop, max_crop)
        
        # Generate random center point
        center = F.normalize(torch.randn(1, 1, 3), p=2, dim=-1)
        
        # Calculate distances from center to all points
        distance_matrix = torch.norm(center.unsqueeze(2) - xyz.unsqueeze(1), p=2, dim=-1)  # (1, 1, N)
        
        # Sort points by distance (closest first)
        idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0, 0]  # (N,)
        
        if self.padding_zeros:
            # Pad with zeros instead of removing points
            input_data = xyz.clone()
            input_data[0, idx[:num_crop]] = input_data[0, idx[:num_crop]] * 0
        else:
            # Remove the closest points (keep the furthest ones) - this creates the partial cloud
            input_data = xyz.clone()[0, idx[num_crop:]].unsqueeze(0)  # (1, N-num_crop, 3)
        
        # Convert back to numpy and ensure we have target_points
        result = input_data.squeeze(0).numpy()
        
        # If we have more points than target, randomly sample
        if result.shape[0] > self.target_points:
            choice = np.random.permutation(result.shape[0])
            result = result[choice[:self.target_points]]
        # If we have fewer points than target, pad with zeros
        elif result.shape[0] < self.target_points:
            zeros = np.zeros((self.target_points - result.shape[0], 3))
            result = np.concatenate([result, zeros])
            
        return result

class Downsample:
    def __init__(self, n_points: int = 2048):
        self.n_points = n_points
    
    def __call__(self, point_cloud: np.ndarray) -> np.ndarray:
        choice = np.random.permutation(point_cloud.shape[0])
        point_cloud = point_cloud[choice[:self.n_points]]
        if point_cloud.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - point_cloud.shape[0], 3))
            point_cloud = np.concatenate([point_cloud, zeros])

        return point_cloud

class ToTensor:
    def __call__(self, point_cloud: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(point_cloud).float()


class Compose:
    """
    Composes several transforms together.
    
    This class allows you to chain multiple transformations together,
    applying them sequentially to the input data.
    """
    
    def __init__(self, transforms):
        """
        Initialize the Compose transform.
        
        Args:
            transforms: List of transform objects to apply in sequence
        """
        self.transforms = transforms
    
    def __call__(self, data):
        """
        Apply all transforms in sequence to the input data.
        
        Args:
            data: Input data (typically a point cloud)
            
        Returns:
            Transformed data after applying all transforms in sequence
        """
        for transform in self.transforms:
            data = transform(data)
        return data
    

class SeparatePointCloud:
    """Separate point cloud: usage: using to generate the incomplete point cloud with a setted number."""
    
    def __init__(self, num_points: int = 8192, padding_zeros: bool = False):
        """
        Initialize the separate point cloud transform.
        
        Args:
            num_points: Number of points in the complete point cloud (default: 8192)
            padding_zeros: Whether to pad with zeros instead of removing points (default: False)
        """
        self.num_points = num_points
        self.padding_zeros = padding_zeros
    
    def __call__(self, xyz: torch.Tensor, crop: int | list[int], fixed_points: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Separate point cloud into input and crop data.
        
        Args:
            xyz: Input point cloud of shape (B, N, 3)
            crop: Number of points to crop (can be int or list [min, max])
            fixed_points: Fixed center points to use (optional)
            
        Returns:
            tuple: (input_data, crop_data) where both are torch.Tensors
        """
        batch_size, n, c = xyz.shape
        
        assert n == self.num_points, f"Expected {self.num_points} points, got {n}"
        assert c == 3, f"Expected 3 coordinates, got {c}"
        
        if crop == self.num_points:
            return xyz, torch.empty(0, 3)  # Return empty tensor instead of None
        
        INPUT = []
        CROP = []
        
        for points in xyz:
            if isinstance(crop, list):
                num_crop = random.randint(crop[0], crop[1])
            else:
                num_crop = crop
            
            points = points.unsqueeze(0)  # (1, N, 3)
            
            if fixed_points is None:
                center = F.normalize(torch.randn(1, 1, 3), p=2, dim=-1)
            else:
                if isinstance(fixed_points, list):
                    fixed_point = random.sample(fixed_points, 1)[0]
                else:
                    fixed_point = fixed_points
                center = fixed_point.reshape(1, 1, 3)
            
            # Calculate distance matrix
            distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p=2, dim=-1)  # (1, 1, N)
            
            # Sort by distance (closest first)
            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0, 0]  # (N,)
            
            if self.padding_zeros:
                input_data = points.clone()
                input_data[0, idx[:num_crop]] = input_data[0, idx[:num_crop]] * 0
            else:
                input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0)  # (1, N-num_crop, 3)
            
            crop_data = points.clone()[0, idx[:num_crop]].unsqueeze(0)  # (1, num_crop, 3)
            
            if isinstance(crop, list):
                # Apply FPS if crop is a list (you'll need to implement fps function)
                INPUT.append(input_data)  # For now, just append without FPS
                CROP.append(crop_data)    # For now, just append without FPS
            else:
                INPUT.append(input_data)
                CROP.append(crop_data)
        
        input_data = torch.cat(INPUT, dim=0)  # (B, N', 3)
        crop_data = torch.cat(CROP, dim=0)    # (B, M, 3)
        
        return input_data.contiguous(), crop_data.contiguous()
    

