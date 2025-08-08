import numpy as np
import open3d as o3d

class ViewpointSampler:
    def __init__(self, strategy='cube', n_viewpoints=None, distance=1.0):
        self.strategy = strategy
        self.n_viewpoints = n_viewpoints
        self.distance = distance
    
    def sample_viewpoints(self):
        if self.strategy == 'cube':
            return self._sample_cube_viewpoints()
        elif self.strategy == 'sphere':
            if self.n_viewpoints is None:
                raise ValueError("n_viewpoints must be specified for sphere strategy")
            return self._sample_sphere_viewpoints()
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}. Use 'cube' or 'sphere'")
        
    def _sample_sphere_viewpoints(self):
        """Generate random viewpoints on a sphere surface."""
        viewpoints = []
        
        for _ in range(self.n_viewpoints):
            # Generate random point on unit sphere
            point = np.random.randn(3)
            point = point / np.linalg.norm(point)
            viewpoints.append(point * self.distance)
        
        return np.array(viewpoints, dtype=np.float32)
    
    def _sample_cube_viewpoints(self):
        """Generate viewpoints at the 8 corners of a cube."""
        corners = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    corners.append([x * self.distance, y * self.distance, z * self.distance])
        return np.array(corners, dtype=np.float32)
    
class VirtualScanner:
    def __init__(self, mesh, strategy='cube', n_viewpoints=None, distance=2.0, n_rays=5000, fov_degrees=60, n_points=2048):
        """
        - mesh(o3d.geometry.TriangleMesh): the mesh to scan
        - strategy('cube' or 'sphere'): the strategy to generate viewpoints
        - n_viewpoints(int): number of viewpoints for sphere strategy
        - distance(float): if strategy is 'cube', the edge length of the cube, otherwise, radius of the sphere
        - n_rays(int): number of rays for each viewpoint (reduced from 10000 to 5000 for better distribution)
        - fov_degrees(float): field of view in degrees (reduced from 90 to 60 for less clustering)
        - n_points(int): number of points to sample for each viewpoint
        """
        self.mesh = mesh
        self.n_viewpoints = n_viewpoints
        self.strategy = strategy
        self.distance = distance
        self.n_rays = n_rays
        self.fov_radians = np.radians(fov_degrees)
        self.viewpoints = ViewpointSampler(strategy, n_viewpoints, distance).sample_viewpoints()
        self.n_points = n_points
        
    def process(self):
        all_points = []
        all_normals = []
        
        
        for i, viewpoint in enumerate(self.viewpoints):
            # Create a virtual camera at the viewpoint
            camera_points, camera_normals = self.scan(self.mesh, viewpoint, self.n_rays)
            
            if camera_points is not None and len(camera_points) > 0:
                all_points.append(camera_points)
                all_normals.append(camera_normals)
        
        return self.viewpoints, all_points, all_normals
    
    def scan(self, mesh, viewpoint, n_rays=10000):
        """Simulate scanning from a specific viewpoint using ray casting."""
        camera_position = viewpoint
        look_at = np.array([0.0, 0.0, 0.0])
        
        forward = look_at - camera_position
        forward = forward / np.linalg.norm(forward)

        up = np.array([0, 1, 0])
        if abs(np.dot(forward, up)) > 0.9:
            up = np.array([0, 0, 1])
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        R = np.column_stack([right, up, forward])
        
        points = []
        normals = []
        
        fov = self.fov_radians
        
        all_rays = []
        for i in range(n_rays):
            # Simple uniform cone sampling
            # Sample angle from center axis (0 to fov/2) using cosine distribution for uniform coverage
            cos_angle = np.random.uniform(np.cos(fov/2), 1.0)
            angle = np.arccos(cos_angle)
            azimuth = np.random.uniform(0, 2 * np.pi)
            
            local_direction = np.array([
                np.sin(angle) * np.cos(azimuth),
                np.sin(angle) * np.sin(azimuth),
                np.cos(angle)
            ])
            
            ray_direction = R @ local_direction
            ray_direction = ray_direction / np.linalg.norm(ray_direction)
            
            ray_data = np.concatenate([camera_position, ray_direction])
            all_rays.append(ray_data)
        
        # Cast all rays at once
        rays_tensor = o3d.core.Tensor(all_rays, dtype=o3d.core.Dtype.Float32)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        
        ans = scene.cast_rays(rays_tensor)
        
        t_hit = ans['t_hit'].numpy()
        hit_mask = t_hit < float('inf')
        
        max_ray_length = 10.0
        hit_mask = hit_mask & (t_hit < max_ray_length)
        
        for i in range(n_rays):
            if hit_mask[i]:
                # Hit point
                ray_origin = camera_position
                ray_direction = all_rays[i][3:6]  # Extract direction from ray data
                hit_point = ray_origin + ray_direction * t_hit[i]
                points.append(hit_point)
                normals.append(ray_direction)
        
        if len(points) == 0:
            points = np.zeros((self.n_points, 3), dtype=np.float32)
            normals = np.zeros((self.n_points, 3), dtype=np.float32)
            return points, normals
        
        points = np.array(points, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)
        
        # Remove duplicate points (within small tolerance)
        if len(points) > 1:
            # Use simple distance-based deduplication
            from scipy.spatial.distance import pdist, squareform
            distances = squareform(pdist(points))
            np.fill_diagonal(distances, np.inf)
            
            # Find points that are too close to each other
            min_distance = 0.01  # Minimum distance between points
            too_close = np.any(distances < min_distance, axis=1)
            
            # Keep only unique points
            unique_mask = ~too_close
            points = points[unique_mask]
            normals = normals[unique_mask]
        
        # Sample or pad to target number of points
        if points.shape[0] > self.n_points:
            # Random sampling without replacement
            choice = np.random.choice(points.shape[0], self.n_points, replace=False)
            points = points[choice]
            normals = normals[choice]
        elif points.shape[0] < self.n_points:
            # Pad with zeros
            zeros = np.zeros((self.n_points - points.shape[0], 3), dtype=np.float32)
            points = np.concatenate([points, zeros])
            normals = np.concatenate([normals, zeros])
        
        return points, normals
    
    
if __name__ == "__main__":
    from tools.data.ABCReader import ABCReader
    from tools.data.SolidProcessor import SolidProcessor
    import trimesh
    import os
    
    for folder_name in os.listdir("D:/XiaoLunWen/data/abc_data"):
        reader = ABCReader()
        file_name = os.listdir(f"D:/XiaoLunWen/data/abc_data/{folder_name}")[0]
        reader.read_step_file(f"D:/XiaoLunWen/data/abc_data/{folder_name}/{file_name}")
        if reader.num_solids == 0:
            continue
        elif reader.num_solids > 1:
            print(f"Multiple solids found in {folder_name}")
            continue
        shape = reader.get_solids()[0]
        solid_processor = SolidProcessor(shape)
        solid_processor.normalize_shape()
        solid_processor.export_mesh(f"mesh.stl")

        
        scanner = VirtualScanner(solid_processor.mesh, strategy='sphere', n_viewpoints=64, distance=2.0, n_rays=5000, fov_degrees=60, n_points=2048)
        viewpoints, points, normals = scanner.process()
        
        for i in range(len(points)):
            trimesh.PointCloud(points[i].reshape(-1, 3)).export(f"{file_name}_{i}.ply")
        break
    