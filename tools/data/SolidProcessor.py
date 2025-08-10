import open3d as o3d
import numpy as np
import os

from tools.data import RandomCropper, VirtualScanner, PhotoRenderer

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

class SolidProcessor:
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
                    f.append([t.Value(3) + cur_vertex_size - 1, 
                              t.Value(2) + cur_vertex_size - 1,
                              t.Value(1) + cur_vertex_size - 1])
                else:
                    f.append([t.Value(1) + cur_vertex_size - 1, 
                              t.Value(2) + cur_vertex_size - 1,
                              t.Value(3) + cur_vertex_size - 1])
        return v, f
    
    #########################
    # Data export functions #
    #########################
    
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
        # self._create_dir_if_not_exist(os.path.dirname(mesh_path))
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
    
    def export_point_cloud_numpy(self, point_cloud_path):
        self._create_dir_if_not_exist(os.path.dirname(point_cloud_path))
        points = np.asarray(self.point_cloud.points, dtype=np.float32)
        np.savez_compressed(point_cloud_path, points=points)

    def export_scanned_point_cloud(self, scanned_pc_path, n_points=2048, strategy='sphere', n_viewpoints=64, radius=2.0, n_rays=2048):
        """
        Mimic a real-world scanner by generating point clouds from multiple viewpoints.
        
        Args:
            scanned_pc_path: Output path for npz file
            n_points: Number of points to sample (default: 2048)
            n_viewpoints: Number of viewpoints to use (default: 8)
            strategy: Viewpoint selection strategy ('sphere' or 'cube')
            radius: Radius of the sphere or half-edge length of cube (default: 1.0)
        """
        self._create_dir_if_not_exist(os.path.dirname(scanned_pc_path))
        
        virtual_scanner = VirtualScanner(self.mesh, strategy, n_viewpoints, radius, n_rays, n_points)
        # Ensure we have a mesh
        if self.mesh is None:
            vertices, faces = self.get_triangulations()
            self.mesh = o3d.geometry.TriangleMesh()
            self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
            self.mesh.triangles = o3d.utility.Vector3iVector(faces)
            self.mesh.compute_vertex_normals()
        
        viewpoints, all_points, all_normals = virtual_scanner.process()
        
        assert len(all_points) == len(all_normals) == len(viewpoints)
        
        import h5py
        with h5py.File(scanned_pc_path, 'w') as f:
            f.create_dataset('points', data=all_points, compression='gzip', compression_opts=9)
            f.create_dataset('normals', data=all_normals, compression='gzip', compression_opts=9)
            f.create_dataset('viewpoints', data=viewpoints, compression='gzip', compression_opts=9)

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
        cropper = RandomCropper(size=size, n_points=n_points, padding_mode=padding_mode)
        input_samples, crop_samples = cropper.process(points)
        
        # Save in HDF5 format
        with h5py.File(cropped_pc_path, 'w') as f:
            f.create_dataset('input_data', data=input_samples, compression='gzip', compression_opts=9)
            f.create_dataset('crop_data', data=crop_samples, compression='gzip', compression_opts=9)
            f.attrs['n_points'] = n_points
            f.attrs['padding_mode'] = padding_mode
            f.attrs['num_samples'] = size
        
    def export_uv_grids(self, uv_grid_path, sample_resolution=16):
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
        np.savez_compressed(uv_grid_path, 
                            face_sample_points=face_sample_points)
    
    def export_photos(self, photo_path):
        renderer = PhotoRenderer(self.solid)
        svr_images, mvr_images = renderer.process()
        np.savez_compressed(photo_path, svr=svr_images, mvr=mvr_images)
    
    def _create_dir_if_not_exist(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
    