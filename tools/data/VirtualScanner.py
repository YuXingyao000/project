import numpy as np
import open3d as o3d
import cv2

class ViewpointSampler:
    def __init__(self, strategy='cube', n_viewpoints=None, distance=1.0):
        self.strategy = strategy
        self.n_viewpoints = n_viewpoints
        self.distance = distance

    def sample_viewpoints(self):
        if self.strategy == 'cube':
            return self._sample_cube_corners()
        elif self.strategy == 'sphere':
            if self.n_viewpoints is None:
                raise ValueError("n_viewpoints must be specified for sphere strategy")
            return self._sample_sphere_viewpoints()
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}. Use 'cube' or 'sphere'")

    def _sample_sphere_viewpoints(self):
        """Generate viewpoints evenly distributed on a sphere (Fibonacci sphere)."""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))
        for i in range(self.n_viewpoints):
            y = 1 - (i / float(self.n_viewpoints - 1)) * 2
            r = np.sqrt(max(0, 1 - y * y))
            theta = phi * i
            x = np.cos(theta) * r
            z = np.sin(theta) * r
            points.append([self.distance * x, self.distance * y, self.distance * z])
        return np.array(points, dtype=np.float32)

    def _sample_cube_corners(self):
        """Generate viewpoints at the 8 corners of a cube."""
        corners = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    corners.append([x * self.distance, y * self.distance, z * self.distance])
        return np.array(corners, dtype=np.float32)


class VirtualScannerBackProjection:
    def __init__(self, mesh, strategy='cube', n_viewpoints=64, distance=2.0,
                 width=640, height=480, fov_degrees=60, n_points=10000):
        """
        - mesh (o3d.geometry.TriangleMesh): target mesh
        - strategy ('cube' | 'sphere'): viewpoint strategy
        - n_viewpoints (int): number of viewpoints if 'sphere'
        - distance (float): camera distance from origin
        - width, height (int): image resolution for depth rendering
        - fov_degrees (float): camera field of view
        - n_points (int): number of points sampled per viewpoint
        """
        self.mesh = mesh
        self.strategy = strategy
        self.n_viewpoints = n_viewpoints
        self.distance = distance
        self.width = width
        self.height = height
        self.fov = np.deg2rad(fov_degrees)
        self.n_points = n_points

        # Generate viewpoints via sampler
        self.viewpoints = ViewpointSampler(strategy, n_viewpoints, distance).sample_viewpoints()

        # Set intrinsics (simple pinhole)
        fx = fy = (0.5 * width) / np.tan(self.fov / 2)
        cx, cy = width / 2, height / 2
        self.intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float32)

        # Build raycasting scene
        self.scene = o3d.t.geometry.RaycastingScene()
        _ = self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def _extrinsic_from_viewpoint(self, cam_pos):
        # Look at origin
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0, 1, 0], dtype=np.float32)
        if abs(np.dot(forward, up)) > 0.9:
            up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        R = np.stack([right, up, forward], axis=1)
        T = -R.T @ cam_pos
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = R.T
        extrinsic[:3, 3] = T
        return extrinsic

    def _render_depth(self, extrinsic):
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=o3d.core.Tensor(self.intrinsic),
            extrinsic_matrix=o3d.core.Tensor(extrinsic),
            width_px=self.width,
            height_px=self.height
        )
        ans = self.scene.cast_rays(rays)
        depth = ans['t_hit'].numpy().reshape(self.height, self.width)
        normals = ans['primitive_normals'].numpy().reshape(self.height, self.width, 3)
        return depth, normals

    def _backproject(self, depth, extrinsic, normals):
        h, w = depth.shape
        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        cx, cy = self.intrinsic[0, 2], self.intrinsic[1, 2]

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        Z = depth
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        ones = np.ones_like(Z)

        pts_cam = np.stack([X, Y, Z, ones], axis=-1).reshape(-1, 4)
        mask = np.isfinite(Z).reshape(-1)

        # Transform points to world space
        pts_world = (extrinsic @ pts_cam.T).T[:, :3]

        # Transform normals (rotation only, no translation)
        R = extrinsic[:3, :3]
        normals_flat = normals.reshape(-1, 3)
        normals_world = (R @ normals_flat.T).T

        result_pc = pts_world[mask]
        result_normals = normals_world[mask]
        assert result_pc.shape == result_normals.shape
        
        return np.hstack([result_pc, result_normals])

    def save_depth_as_image(depth, path="depth.png"):
        depth_vis = depth.copy()
        depth_vis[np.isinf(depth_vis)] = 0  # set inf to 0
        depth_vis[np.isnan(depth_vis)] = 0  # set NaN to 0

        # Normalize to 0–255 for visualization
        if depth_vis.max() > 0:
            depth_vis = depth_vis / depth_vis.max()
        depth_vis = (depth_vis * 255).astype(np.uint8)

        cv2.imwrite(path, depth_vis)
    
    def process(self):
        all_points = []
        for vp in self.viewpoints:
            extr = self._extrinsic_from_viewpoint(vp)
            depth, normals = self._render_depth(extr)
            pts = self._backproject(depth, extr, normals)

            # Sample or pad
            if pts.shape[0] > self.n_points:
                choice = np.random.choice(pts.shape[0], self.n_points, replace=False)
                pts = pts[choice]
            elif pts.shape[0] < self.n_points:
                pad = np.zeros((self.n_points - pts.shape[0], 3), dtype=np.float32)
                pts = np.vstack([pts, pad])

            all_points.append(pts.astype(np.float32))

        return self.viewpoints, np.array(all_points, dtype=np.float32)
