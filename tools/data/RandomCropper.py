import random
import numpy as np

class RandomCropper:
    def __init__(self, size=64, n_points=2048, padding_mode='zero'):
        self.size = size
        self.n_points = n_points
        self.padding_mode = padding_mode

    def process(self, point_cloud: np.ndarray):
        input_samples = []
        crop_samples = []
        for i in range(self.size):
            input_data, crop_data = self.random_crop(point_cloud)
            input_samples.append(input_data)
            crop_samples.append(crop_data)
        input_samples = np.stack(input_samples, axis=0)  # Shape: (size, n_points, 3)
        crop_samples = np.stack(crop_samples, axis=0)    # Shape: (size, n_points, 3)
        return input_samples, crop_samples
    
    def random_crop(self, xyz: np.ndarray):
        """
        Randomly crop a point cloud into input and crop data with downsampling.
        
        Args:
            xyz: Input point cloud of shape (N, 3) where N=8192
            n_points: Number of points to downsample to (default: 2048)
            padding_mode: Padding mode when downsampling ('zero' or 'random')
            fixed_points: Fixed center points to use (optional), shape (M, 3)
            
        Returns:
            tuple: (input_data, crop_data) where both are numpy arrays
        """
        if xyz.shape[0] != 8192:
            raise ValueError(f"Expected point cloud with 8192 points, got {xyz.shape[0]}")
        
        # Randomly sample crop ratio between 0.25 and 0.75
        crop_ratio = random.uniform(0.25, 0.75)
        
        num_crop = int(8192 * crop_ratio)
        num_input = 8192 - num_crop
        
        # Select center point for cropping
        # Random direction from unit sphere (following dataset/transform.py approach)
        center = np.random.randn(1, 3)
        center = center / np.linalg.norm(center, axis=1, keepdims=True)  # Normalize to unit length
        
        # Calculate distances from center to all points
        distances = np.linalg.norm(xyz - center, axis=1)  # Shape: (8192,)
        
        # Sort points by distance (closest first)
        sorted_indices = np.argsort(distances)
        
        # Split into crop and input points
        crop_indices = sorted_indices[:num_crop]
        input_indices = sorted_indices[num_crop:]
        
        # Extract the point sets
        crop_data = xyz[crop_indices]  # Shape: (num_crop, 3)
        input_data = xyz[input_indices]  # Shape: (num_input, 3)
        
        # Downsample and pad the cropped point cloud
        if crop_data.shape[0] > self.n_points:
            # Randomly sample n_points from crop_data
            choice = np.random.permutation(crop_data.shape[0])
            crop_data = crop_data[choice[:self.n_points]]
        elif crop_data.shape[0] < self.n_points:
            # Pad with zeros or random points
            if self.padding_mode == 'zero':
                zeros = np.zeros((self.n_points - crop_data.shape[0], 3))
                crop_data = np.concatenate([crop_data, zeros])
            elif self.padding_mode == 'random':
                # Generate random points within the bounds of the original point cloud
                bounds_min = xyz.min(axis=0)
                bounds_max = xyz.max(axis=0)
                random_points = np.random.uniform(bounds_min, bounds_max, (self.n_points - crop_data.shape[0], 3))
                crop_data = np.concatenate([crop_data, random_points])
            else:
                raise ValueError(f"Invalid padding_mode: {self.padding_mode}. Use 'zero' or 'random'")
        
        # Downsample and pad the input point cloud
        if input_data.shape[0] > self.n_points:
            # Randomly sample n_points from input_data
            choice = np.random.permutation(input_data.shape[0])
            input_data = input_data[choice[:self.n_points]]
        elif input_data.shape[0] < self.n_points:
            # Pad with zeros or random points
            if self.padding_mode == 'zero':
                zeros = np.zeros((self.n_points - input_data.shape[0], 3))
                input_data = np.concatenate([input_data, zeros])
            elif self.padding_mode == 'random':
                # Generate random points within the bounds of the original point cloud
                bounds_min = xyz.min(axis=0)
                bounds_max = xyz.max(axis=0)
                random_points = np.random.uniform(bounds_min, bounds_max, (self.n_points - input_data.shape[0], 3))
                input_data = np.concatenate([input_data, random_points])
            else:
                raise ValueError(f"Invalid padding_mode: {self.padding_mode}. Use 'zero' or 'random'")
        
        return input_data, crop_data