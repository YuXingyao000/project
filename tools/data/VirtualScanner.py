import numpy as np
import open3d as o3d

from tools.data.ViewpointSampler import ViewpointSampler

class VirtualScanner:
    def __init__(self, mesh, strategy='cube', n_viewpoints=None, distance=1.0, n_rays=2048):
        """
        - mesh(o3d.geometry.TriangleMesh): the mesh to scan
        - strategy('cube' or 'sphere'): the strategy to generate viewpoints
        - n_viewpoints(int): number of viewpoints for sphere strategy
        - distance(float): if strategy is 'cube', the edge length of the cube, otherwise, radius of the sphere
        - n_rays(int): number of rays for each viewpoint
        """
        self.mesh = mesh
        self.n_viewpoints = n_viewpoints
        self.strategy = strategy
        self.distance = distance
        self.n_rays = n_rays
        self.viewpoints = ViewpointSampler(strategy, n_viewpoints, distance).sample_viewpoints()
        
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
    
    def scan(self, mesh, viewpoint, n_rays=2048):
        """Simulate scanning from a specific viewpoint using ray casting."""
        # Convert viewpoint to camera parameters
        camera_position = viewpoint
        look_at = np.array([0.0, 0.0, 0.0])  # Look at origin
        up = np.array([0.0, 1.0, 0.0])  # Up direction
        
        # Calculate camera orientation
        forward = look_at - camera_position
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Create virtual camera
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        camera.translate(camera_position)
        
        # Simple ray casting simulation
        # For each ray direction, find intersection with mesh
        points = []
        normals = []
        
        # Generate ray directions in a cone towards the object
        fov = np.pi / 2  # 90 degree field of view
        
        for _ in range(n_rays):
            # Proper cone sampling using spherical coordinates
            # Sample angle from center axis (0 to fov)
            angle = np.random.uniform(0, fov / 2)
            # Sample azimuth around the cone (0 to 2π)
            azimuth = np.random.uniform(0, 2 * np.pi)
            
            # Convert to unit vector in scanner's local coordinate system
            # where z-axis points toward the object
            local_direction = np.array([
                np.sin(angle) * np.cos(azimuth),
                np.sin(angle) * np.sin(azimuth),
                np.cos(angle)
            ])
            
            # Transform from local scanner coordinates to world coordinates
            # Build rotation matrix from scanner's coordinate system
            forward = -camera_position / np.linalg.norm(camera_position)  # Toward object
            
            # Robust construction of orthogonal coordinate system
            # Use Gram-Schmidt process to ensure orthogonality
            up = np.array([0, 1, 0])  # Up direction along positive Y-axis
            
            # Check if forward and up are nearly parallel
            if abs(np.dot(forward, up)) > 0.9:
                # If nearly parallel, use a different initial up
                up = np.array([0, 0, 1])  # Fallback to Z-axis
            
            # Gram-Schmidt orthogonalization
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            
            # Recompute up to ensure perfect orthogonality
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)
            
            # Rotation matrix: [right, up, forward]
            R = np.column_stack([right, up, forward])
            
            # Transform local direction to world coordinates
            ray_direction = R @ local_direction
            
            # Ray casting
            ray_origin = camera_position
            ray_end = camera_position + ray_direction * 10  # Ray length
            
            # Use Open3D's ray casting
            ray = o3d.core.Tensor([ray_origin, ray_end], dtype=o3d.core.Dtype.Float32)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            
            ans = scene.cast_rays(ray)
            
            if ans['t_hit'].numpy()[0] < float('inf'):
                # Hit point
                hit_point = ray_origin + ray_direction * ans['t_hit'].numpy()[0]
                points.append(hit_point)
                
                # Get normal at hit point (simplified)
                # In a real implementation, you'd interpolate the normal
                normals.append(ray_direction)  # Simplified normal
        
        # Pad points and normals to length n_rays with zeros if needed
        points_arr = np.array(points, dtype=np.float32)
        normals_arr = np.array(normals, dtype=np.float32)
        n_points = len(points_arr)
        if n_points < n_rays:
            pad_points = np.zeros((n_rays - n_points, 3), dtype=np.float32)
            pad_normals = np.zeros((n_rays - n_points, 3), dtype=np.float32)
            points_arr = np.vstack([points_arr, pad_points])
            normals_arr = np.vstack([normals_arr, pad_normals])
        elif n_points > n_rays:
            points_arr = points_arr[:n_rays]
            normals_arr = normals_arr[:n_rays]
        return points_arr, normals_arr
    
    

    