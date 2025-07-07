import open3d as o3d
import numpy as np
import random


class PartialPointCloudSampler:
    def __init__(self, point_cloud_path):
        point_cloud = o3d.io.read_point_cloud(point_cloud_path)
        self.point_cloud = np.asarray(point_cloud.points)

    @staticmethod
    def generate_viewpoints(radius=2, elevations=[30, -30], num_azimuths=8):
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
    
    def sample_partial_point_cloud(self, viewpoint, n_points=None):
        """
        Sample partial point cloud based on evaluation pipeline:
        1. Choose a viewpoint from 8 viewpoints
        2. Randomly choose n as 2048, 4096 or 6144
        3. Remove n furthest points from the viewpoint
        4. Downsample the remaining point clouds to 2048 points
        
        Args:
            viewpoint: (x, y, z) camera position
            n_points: number of points to remove (if None, randomly choose from [2048, 4096, 6144])
        
        Returns:
            numpy array of shape (2048, 3) - downsampled partial point cloud
        """
        
        # Step 2: Randomly choose n as 2048, 4096 or 6144 if not provided
        if n_points is None:
            n_points = random.choice([2048, 4096, 6144])
        
        # Step 3: Remove n furthest points from the viewpoint
        # Calculate distances from viewpoint to all points
        distances = np.linalg.norm(self.point_cloud - viewpoint, axis=1)
        
        # Sort points by distance (closest first)
        sorted_indices = np.argsort(distances)
        
        # Keep the closest points (remove the furthest n_points)
        remaining_indices = sorted_indices[:-n_points]
        remaining_points = self.point_cloud[remaining_indices]
        
        # Step 4: Downsample the remaining point clouds to 2048 points
        if len(remaining_points) > 2048:
            # Randomly sample 2048 points
            random_indices = np.random.choice(len(remaining_points), 2048, replace=False)
            partial_point_cloud = remaining_points[random_indices]
        else:
            # If we have fewer than 2048 points, pad with zeros or repeat points
            if len(remaining_points) == 0:
                # If no points remain, return zeros
                partial_point_cloud = np.zeros((2048, 3))
            else:
                # Repeat points to reach 2048
                repeat_times = 2048 // len(remaining_points)
                remainder = 2048 % len(remaining_points)
                partial_point_cloud = np.tile(remaining_points, (repeat_times, 1))
                if remainder > 0:
                    partial_point_cloud = np.vstack([partial_point_cloud, remaining_points[:remainder]])
        
        return partial_point_cloud