import open3d as o3d
import numpy as np
import os

from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord  # type: ignore

class PointCloudSampler:
    def __init__(self, mesh) -> None:
        self.mesh = mesh
        self.mesh.compute_vertex_normals()

        self.point_cloud = None
        self.viewpoints = None
    
    def sample_points(self, gt_number=8192, method="uniform"):
        """
        Sample points on the mesh surface using Open3D's built-in methods.
        
        Args:
            gt_number: Target number of points to sample
            method: Sampling method - "uniform" or "poisson"
        """
        if method == "uniform":
            # Uniform sampling - area-weighted distribution
            pcd = self.mesh.sample_points_uniformly(number_of_points=gt_number, use_triangle_normal=True)
        elif method == "poisson":
            # Poisson disk sampling - more uniform distribution
            pcd = self.mesh.sample_points_poisson_disk(number_of_points=gt_number, use_triangle_normal=True)
        else:
            raise ValueError("Method must be 'uniform' or 'poisson'")
        
        self.point_cloud = np.asarray(pcd.points)
        return self.point_cloud
       
    def generate_viewpoints(self, radius=2, elevations=[30, -30], num_azimuths=8):
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
    
    
    def export_point_cloud(self, output_path, sample_num=8196, sample_method="uniform"):
        """
        Export the sampled point cloud.
        """
        
        if sample_method == "uniform":
            pcd = self.mesh.sample_points_uniformly(number_of_points=sample_num, use_triangle_normal=True)
        elif sample_method == "poisson":
            pcd = self.mesh.sample_points_poisson_disk(number_of_points=sample_num, use_triangle_normal=True)
        else:
            raise ValueError("Method must be 'uniform' or 'poisson'")
        
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Point cloud exported to: {output_path}")
