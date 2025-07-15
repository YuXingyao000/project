import numpy as np
import torch


class Partial:
    """Transformations for point cloud data during training."""
    
    def __init__(self, target_points: int = 2048, min_ratio: float = 0.25, max_ratio: float = 0.75, radius=2, elevations=[30, -30], num_azimuths=8):
        """
        Initialize the point cloud transform.
        Generate viewpoints and sample partial point cloud.
        Args:
            target_points: Number of points to downsample to (default: 2048)
            min_ratio: Minimum ratio of points to sample (default: 0.25)
            max_ratio: Maximum ratio of points to sample (default: 0.75)
            radius: Radius of the viewpoint (default: 2)
            elevations: Elevations of the viewpoint (default: [30, -30])
            num_azimuths: Number of azimuths of the viewpoint (default: 8)
        """
        self.viewpoints = self._generate_viewpoints(radius=radius, elevations=elevations, num_azimuths=num_azimuths)
        self.target_points = target_points
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def _generate_viewpoints(self, radius=2, elevations=[30, -30], num_azimuths=8):
        """
        Generate 16 viewpoints: 2 elevation rings, 8 azimuths each.
        Returns a list of (x, y, z) camera positions.
        """
        viewpoints = []
        for elev in elevations:
            elev_rad = np.deg2rad(elev)
            for i in range(num_azimuths):
                azim = i * 360 / num_azimuths
                azim_rad = np.deg2rad(azim)
                x = radius * np.cos(elev_rad) * np.cos(azim_rad)
                y = radius * np.cos(elev_rad) * np.sin(azim_rad)
                z = radius * np.sin(elev_rad)
                viewpoints.append([x, y, z])
        return np.array(viewpoints)
    
    def __call__(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Apply the transformation to a point cloud.
        
        Args:
            point_cloud: Input point cloud of shape (N, 3) where N >= target_points
            
        Returns:
            Transformed point cloud of shape (target_points, 3)
        """
        # Get the number of points in the input
        num_points = point_cloud.shape[0]
        assert num_points == 8192, f"Point cloud must have 8192 points, but has {num_points} points"
        # Randomly choose n from 2048 to 6144 (25% to 75% of complete point cloud)
        # Assuming complete point cloud has 8192 points (as seen in dataset.py)
        complete_points = 8192
        min_points = int(complete_points * self.min_ratio)  # 2048
        max_points = int(complete_points * self.max_ratio)  # 6144
        
        # Ensure we don't exceed the available points
        max_points = min(max_points, num_points)
        min_points = min(min_points, num_points)
        
        n_points = np.random.randint(min_points, max_points + 1)
        # Randomly select a viewpoint
        viewpoint = self.viewpoints[np.random.randint(0, len(self.viewpoints))]
        # Remove n furthest points from the viewpoint
        distances = np.linalg.norm(point_cloud - viewpoint, axis=1)
        sorted_indices = np.argsort(distances)
        remaining_indices = sorted_indices[:n_points]
        remaining_points = point_cloud[remaining_indices]
        
        assert len(remaining_points) < max_points and len(remaining_points) > min_points, f"Expected {n_points} points, but got {len(remaining_points)} points"
            
        return remaining_points

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
    

