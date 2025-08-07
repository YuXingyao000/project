import numpy as np

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
                    corners.append([x * self.distance / 2.0, y * self.distance / 2.0, z * self.distance / 2.0])
        return np.array(corners, dtype=np.float32)
