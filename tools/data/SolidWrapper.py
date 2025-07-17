import open3d as o3d
import numpy as np
import os
import random

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_REVERSED
from OCC.Core.TopoDS import topods
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core import BRepBndLib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.gp import gp_Trsf, gp_Vec
from OCC.Extend.DataExchange import write_step_file
from OCC.Core.STEPControl import STEPControl_Writer
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface

def extract_primitives(shape, TopoAbs):
    primitives = []
    explorer = TopExp_Explorer(shape, TopoAbs)
    while explorer.More():
        primitives.append(explorer.Current())
        explorer.Next()
    return primitives

class SolidWrapper:
    def __init__(self, solid) -> None:
        self.solid = solid
        self.surfaces = None
        self.curves = None
        self.vertices = None
        self.mesh = None
        self.point_cloud = None
        self.face_sample_points = None
        
    def get_surfaces(self):
        if self.surfaces is None:
            self.surfaces = extract_primitives(self.solid, TopAbs_FACE)
            self.surfaces = [topods.Face(face) for face in self.surfaces]
        return self.surfaces
    
    def get_curves(self):
        if self.curves is None:
            self.curves = extract_primitives(self.solid, TopAbs_EDGE)
            self.curves = [topods.Edge(edge) for edge in self.curves]
        return self.curves
    
    def get_vertices(self):
        if self.vertices is None:
            self.vertices = extract_primitives(self.solid, TopAbs_VERTEX)
            self.vertices = [topods.Vertex(vertex) for vertex in self.vertices]
        return self.vertices
    
    def get_solid(self):
        return self.solid
    
    def normalize_shape(self, bbox_scale=1.0):
        """
        Nomalize a TopoDS_Shape

        Args:
            shape (TopoDS_Shape): Brep shape to be normalized
            bbox_scale (float, optional): Scale factor for the bounding box of the shape. Defaults to 1.0.

        Returns:
            TopoDS_Shape: Normalized Brep shape
        """
        boundingBox = Bnd_Box()
        BRepBndLib.brepbndlib.Add(self.solid, boundingBox)
        xmin, ymin, zmin, xmax, ymax, zmax = boundingBox.Get()
        scale_x = bbox_scale * 2 / (xmax - xmin)
        scale_y = bbox_scale * 2 / (ymax - ymin)
        scale_z = bbox_scale * 2 / (zmax - zmin)
        scaleFactor = min(scale_x, scale_y, scale_z)

        # Translation
        translation_vector = gp_Vec(-(xmax + xmin) / 2, -(ymax + ymin) / 2, -(zmax + zmin) / 2)
        translation_trsf = gp_Trsf()
        translation_trsf.SetTranslationPart(translation_vector)

        # Scale
        scale_trsf = gp_Trsf()
        scale_trsf.SetScaleFactor(scaleFactor)
        scale_trsf.Multiply(translation_trsf)

        transform = BRepBuilderAPI_Transform(scale_trsf)
        transform.Perform(self.solid)
        self.solid = topods.Solid(transform.Shape())
        # Clear cached data
        self.surfaces = None
        self.curves = None
        self.vertices = None
        self.mesh = None
        self.point_cloud = None
        self.face_sample_points = None
    
    def get_triangulations(self, line_deflection=0.1, angle_deflection=0.5):
        if line_deflection > 0:
            mesh = BRepMesh_IncrementalMesh(self.solid, line_deflection, False, angle_deflection)
            mesh.Perform()  # Actually perform the meshing
        v = []
        f = []
        surfaces = self.get_surfaces()
        for surface in surfaces:
            loc = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(surface, loc)
            if triangulation is None:
                print("Ignore surface without triangulation")
                continue
            cur_vertex_size = len(v)
            for i in range(1, triangulation.NbNodes() + 1):
                pnt = triangulation.Node(i)
                v.append([pnt.X(), pnt.Y(), pnt.Z()])
            for i in range(1, triangulation.NbTriangles() + 1):
                t = triangulation.Triangle(i)
                if surface.Orientation() == TopAbs_REVERSED:
                    f.append([t.Value(3) + cur_vertex_size - 1, t.Value(2) + cur_vertex_size - 1,
                              t.Value(1) + cur_vertex_size - 1])
                else:
                    f.append([t.Value(1) + cur_vertex_size - 1, t.Value(2) + cur_vertex_size - 1,
                              t.Value(3) + cur_vertex_size - 1])
        return v, f
    
    def export_solid(self, solid_path):
        self._create_dir_if_not_exist(os.path.dirname(solid_path))
        
        # Try using the lower-level STEP writer
        step_writer = STEPControl_Writer()
        step_writer.Transfer(self.solid, 0)  # 0 = as-is mode
        status = step_writer.Write(str(solid_path))
        
        if status != IFSelect_RetDone:
            print(f"Warning: STEP export may have failed for {solid_path}")
    
    def export_mesh(self, mesh_path, line_deflection=0.1, angle_deflection=0.5):
        """
        Export TopoDS_Solid to STL mesh file
        
        Args:
            mesh_path: Output path for STL file
            line_deflection: Mesh quality parameter (default 0.1)
        """
        self._create_dir_if_not_exist(os.path.dirname(mesh_path))
        vertices, faces = self.get_triangulations(line_deflection, angle_deflection)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # Compute normals
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        self.mesh = mesh
        
    def export_point_cloud(self, point_cloud_path, sample_num=8196, sample_method="uniform"):
        self._create_dir_if_not_exist(os.path.dirname(point_cloud_path))
        if sample_method == "poisson":
            points = self.mesh.sample_points_poisson_disk(sample_num)
        elif sample_method == "uniform":
            points = self.mesh.sample_points_uniformly(sample_num)
        else:
            raise ValueError(f"Invalid sample method: {sample_method}")
        self.point_cloud = points
        o3d.io.write_point_cloud(point_cloud_path, points)
    
    def export_point_cloud_numpy(self, point_cloud_path, sample_num=8196, sample_method="uniform"):
        self._create_dir_if_not_exist(os.path.dirname(point_cloud_path))
        points = np.asarray(self.point_cloud.points, dtype=np.float32)
        np.savez_compressed(point_cloud_path, points=points)

    def export_random_cropped_pc(self, cropped_pc_path, size=64, n_points=2048, padding_mode='zero'):
        """
        Export randomly cropped point clouds in HDF5 format for efficient loading.
        
        Args:
            cropped_pc_path: Output path for HDF5 file
            size: Number of samples to generate
            n_points: Number of points per sample (default: 2048)
            padding_mode: Padding mode ('zero' or 'random')
        """
        import h5py
        
        self._create_dir_if_not_exist(os.path.dirname(cropped_pc_path))
        points = np.asarray(self.point_cloud.points, dtype=np.float32)
        
        # Generate all samples
        input_samples = []
        crop_samples = []
        
        for i in range(size):
            input_data, crop_data = self._random_crop_pc(points, n_points, padding_mode)
            input_samples.append(input_data)
            crop_samples.append(crop_data)
        
        # Convert to numpy arrays
        input_samples = np.stack(input_samples, axis=0)  # Shape: (size, n_points, 3)
        crop_samples = np.stack(crop_samples, axis=0)    # Shape: (size, n_points, 3)
        
        # Save in HDF5 format
        with h5py.File(cropped_pc_path, 'w') as f:
            f.create_dataset('input_data', data=input_samples, compression='gzip', compression_opts=9)
            f.create_dataset('crop_data', data=crop_samples, compression='gzip', compression_opts=9)
            f.attrs['n_points'] = n_points
            f.attrs['padding_mode'] = padding_mode
            f.attrs['num_samples'] = size
        
        print(f"Saved {size} randomly cropped samples to {cropped_pc_path}")
        print(f"Input data shape: {input_samples.shape}")
        print(f"Crop data shape: {crop_samples.shape}")

    def load_random_cropped_pc(self, cropped_pc_path, indices=None):
        """
        Load randomly cropped point clouds from HDF5 file.
        
        Args:
            cropped_pc_path: Path to HDF5 file
            indices: Specific indices to load (None for all)
            
        Returns:
            tuple: (input_data, crop_data) as numpy arrays
        """
        import h5py
        
        with h5py.File(cropped_pc_path, 'r') as f:
            if indices is None:
                input_data = f['input_data'][:]
                crop_data = f['crop_data'][:]
            else:
                input_data = f['input_data'][indices]
                crop_data = f['crop_data'][indices]
        
        return input_data, crop_data
    
    def export_face_sample_points(self, face_sample_points_path, sample_resolution=16):
        face_sample_points = []
        for face in self.get_surfaces():
            surface = BRepAdaptor_Surface(face)
            if surface.GetType() not in [GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                                         GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface]:
                raise ValueError("Unsupported surface type: {}".format(surface.GetType()))
            first_u = surface.FirstUParameter()
            last_u = surface.LastUParameter()
            first_v = surface.FirstVParameter()
            last_v = surface.LastVParameter()
            u = np.linspace(first_u, last_u, num=sample_resolution)
            v = np.linspace(first_v, last_v, num=sample_resolution)
            u, v = np.meshgrid(u, v)
            points = []
            for i in range(u.shape[0]):
                for j in range(u.shape[1]):
                    pnt = surface.Value(u[i, j], v[i, j])
                    props = GeomLProp_SLProps(surface.Surface().Surface(), u[i, j], v[i, j], 1, 0.01)
                    dir = props.Normal()
                    points.append(np.array([pnt.X(), pnt.Y(), pnt.Z(), dir.X(), dir.Y(), dir.Z()], dtype=np.float32))
            face_sample_points.append(np.stack(points, axis=0).reshape(sample_resolution, sample_resolution, -1))
        face_sample_points = np.stack(face_sample_points, axis=0)
        np.savez_compressed(face_sample_points_path, 
                            face_sample_points=face_sample_points)
    
    def _create_dir_if_not_exist(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    def _random_crop_pc(self, xyz: np.ndarray, n_points: int = 2048, padding_mode: str = 'zero', fixed_points: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
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
        import random
        
        if xyz.shape[0] != 8192:
            raise ValueError(f"Expected point cloud with 8192 points, got {xyz.shape[0]}")
        
        # Randomly sample crop ratio between 0.25 and 0.75
        crop_ratio = random.uniform(0.25, 0.75)
        
        num_crop = int(8192 * crop_ratio)
        num_input = 8192 - num_crop
        
        # Select center point for cropping
        if fixed_points is None:
            # Random direction from unit sphere (following dataset/transform.py approach)
            center = np.random.randn(1, 3)
            center = center / np.linalg.norm(center, axis=1, keepdims=True)  # Normalize to unit length
        else:
            # Use provided fixed points
            if fixed_points.ndim == 1:
                center = fixed_points.reshape(1, 3)
            else:
                # Randomly select one from multiple fixed points
                center_idx = random.randint(0, fixed_points.shape[0] - 1)
                center = fixed_points[center_idx:center_idx+1]  # Shape: (1, 3)
        
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
        if crop_data.shape[0] > n_points:
            # Randomly sample n_points from crop_data
            choice = np.random.permutation(crop_data.shape[0])
            crop_data = crop_data[choice[:n_points]]
        elif crop_data.shape[0] < n_points:
            # Pad with zeros or random points
            if padding_mode == 'zero':
                zeros = np.zeros((n_points - crop_data.shape[0], 3))
                crop_data = np.concatenate([crop_data, zeros])
            elif padding_mode == 'random':
                # Generate random points within the bounds of the original point cloud
                bounds_min = xyz.min(axis=0)
                bounds_max = xyz.max(axis=0)
                random_points = np.random.uniform(bounds_min, bounds_max, (n_points - crop_data.shape[0], 3))
                crop_data = np.concatenate([crop_data, random_points])
            else:
                raise ValueError(f"Invalid padding_mode: {padding_mode}. Use 'zero' or 'random'")
        
        # Downsample and pad the input point cloud
        if input_data.shape[0] > n_points:
            # Randomly sample n_points from input_data
            choice = np.random.permutation(input_data.shape[0])
            input_data = input_data[choice[:n_points]]
        elif input_data.shape[0] < n_points:
            # Pad with zeros or random points
            if padding_mode == 'zero':
                zeros = np.zeros((n_points - input_data.shape[0], 3))
                input_data = np.concatenate([input_data, zeros])
            elif padding_mode == 'random':
                # Generate random points within the bounds of the original point cloud
                bounds_min = xyz.min(axis=0)
                bounds_max = xyz.max(axis=0)
                random_points = np.random.uniform(bounds_min, bounds_max, (n_points - input_data.shape[0], 3))
                input_data = np.concatenate([input_data, random_points])
            else:
                raise ValueError(f"Invalid padding_mode: {padding_mode}. Use 'zero' or 'random'")
        
        return input_data, crop_data
    