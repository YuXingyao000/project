import os
import open3d as o3d
import numpy as np
import h5py

from tools.data import RandomCropper, VirtualScannerBackProjection, PhotoRenderer

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_REVERSED
from OCC.Core.TopoDS import topods
from OCC.Core.TopTools import TopTools_HSequenceOfShape
from OCC.Core.TopLoc import TopLoc_Location

from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_MakeEdge
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core import BRepBndLib

from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface, # Face
                              GeomAbs_Circle, GeomAbs_Line, GeomAbs_Ellipse, GeomAbs_BSplineCurve, # Edge
                              GeomAbs_C2) # Continuity
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.gp import gp_Trsf, gp_Vec
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.STEPControl import STEPControl_Writer
from OCC.Core.IFSelect import IFSelect_RetDone


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
        Normalize a TopoDS_Shape so it fits into [-bbox_scale, bbox_scale]^3.
        """
        boundingBox = Bnd_Box()
        BRepBndLib.brepbndlib.Add(self.solid, boundingBox)
        xmin, ymin, zmin, xmax, ymax, zmax = boundingBox.Get()

        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin

        # Scale factor so the largest extent becomes 2 * bbox_scale
        scaleFactor = (bbox_scale * 2) / max(dx, dy, dz)

        # Centering translation
        translation_vector = gp_Vec(-(xmax + xmin) / 2,
                                    -(ymax + ymin) / 2,
                                    -(zmax + zmin) / 2)

        translation_trsf = gp_Trsf()
        translation_trsf.SetTranslationPart(translation_vector)

        # Scale about origin
        scale_trsf = gp_Trsf()
        scale_trsf.SetScaleFactor(scaleFactor)

        # Apply translation then scaling
        trsf = gp_Trsf()
        trsf.Multiply(scale_trsf)
        trsf.Multiply(translation_trsf)

        transform = BRepBuilderAPI_Transform(trsf)
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

    def export_scanned_point_cloud(self, scanned_pc_path, n_points=8196, strategy='sphere', n_viewpoints=64, radius=2.0, n_rays=2048):
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
        
        # Ensure we have a mesh
        if self.mesh is None:
            vertices, faces = self.get_triangulations()
            self.mesh = o3d.geometry.TriangleMesh()
            self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
            self.mesh.triangles = o3d.utility.Vector3iVector(faces)
            self.mesh.compute_vertex_normals()
        virtual_scanner = VirtualScannerBackProjection(self.mesh, strategy, n_viewpoints, n_points=n_points)
        
        viewpoints, all_points = virtual_scanner.process()
        
        assert len(all_points) == len(viewpoints)
        
        np.savez_compressed(
            scanned_pc_path,
            points=np.array(all_points, dtype=np.float32),   # shape: (n_viewpoints, n_points, 6) if normals included
            viewpoints=np.array(viewpoints, dtype=np.float32)
        )

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
        # TODO: Refactor this function. Seperate this function into extract_uv_grids and extract_topology
        face_dict = {}
        for face in self.get_surfaces():
            if face not in face_dict:
                face_dict[face] = len(face_dict)
        
        # Every edge is connected to two faces
        edge_face_look_up_table = {}
        for face in face_dict:
            wires = self._extract_edges_from_faces(face)
            for wire in wires:
                for edge in wire:
                    if edge not in edge_face_look_up_table:
                        edge_face_look_up_table[edge] = [face]
                    else:
                        edge_face_look_up_table[edge].append(face)
        
        num_faces = len(self.get_surfaces())
        face_face_adj_dict = {}
        for edge in edge_face_look_up_table:
            if edge.Reversed() not in edge_face_look_up_table:
                raise ValueError("Edge not in edge_face_look_up_table")
            if len(edge_face_look_up_table[edge]) != 1:
                raise ValueError("Edge indexed by more than 1 faces.")

            index_of_face1 = face_dict[edge_face_look_up_table[edge][0]]
            index_of_face2 = face_dict[edge_face_look_up_table[edge.Reversed()][0]]
            face_pair = (
                index_of_face1,
                index_of_face2
            )

            if face_pair not in face_face_adj_dict:
                face_face_adj_dict[face_pair] = [edge]
            else:
                face_face_adj_dict[face_pair].append(edge)

        # For now we have a dictionary that implies the relationship between intersected faces and edges,
        # we need to merge the edges into a wire for every pair of the intersected faces.
        edge_face_connectivity = []
        for (face1_index, face2_index), edge_list in face_face_adj_dict.items():
            if face1_index == face2_index:  # Skip seam line
                continue

            if len(edge_list) == 1:
                topo_edge = edge_list[0]
            else:
                # TODO: "curve" is "edge" 
                edges_seq = TopTools_HSequenceOfShape()
                for edge in edge_list:
                    edges_seq.Append(edge)
                connected_wire = ShapeAnalysis_FreeBounds.ConnectEdgesToWires(edges_seq, 0.001, False)
                if connected_wire.Length() != 1:
                    raise Exception("Error: Wire creation failed")
                wire = topods.Wire(connected_wire.First())

                # Prepare the control points on a curve to transform the curve into a B-Spline curve
                control_points = []
                wire_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
                while wire_explorer.More():
                    _edge = topods.Edge(wire_explorer.Current())
                    _curve = BRepAdaptor_Curve(_edge)
                    param_start = _curve.FirstParameter() if _edge.Orientation() == 0 else _curve.LastParameter()
                    param_end = _curve.LastParameter() if _edge.Orientation() == 0 else _curve.FirstParameter()
                    param_sampled_on_edge = np.linspace(param_start, param_end, num=sample_resolution)
                    for u in param_sampled_on_edge:
                        control_points.append(_curve.Value(u))
                    wire_explorer.Next()
                # Fit BSpline
                u_points_array = TColgp_Array1OfPnt(1, len(control_points))
                for i in range(len(control_points)):
                    u_points_array.SetValue(i + 1, control_points[i])
                bspline_curve = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 1e-3).Curve()
                topo_edge = BRepBuilderAPI_MakeEdge(bspline_curve).Edge()

            # After merging, we record the edge-face connectivity with edges keeping their topology form.
            # We will transform this connectivity list into the pure index form later.
            edge_face_connectivity.append((
                topo_edge,
                face1_index,
                face2_index
            ))
        
        # Sample faces in the parametric space
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
            points = np.stack(points, axis=0)
            face_center = np.mean(points[:, :3], axis=0)
            face_scale = np.min(np.linalg.norm(points[:, :3] - face_center, axis=1))
            if face_scale < 1e-3:
                raise ValueError("Face scale too small: {}".format(face_scale))
            face_sample_points.append(points.reshape(sample_resolution, sample_resolution, -1))
        face_sample_points = np.stack(face_sample_points, axis=0)
        assert len(face_dict) == num_faces == face_sample_points.shape[0]
        
        # Sample edges in the parametric space
        edge_sample_points = []
        for edge_index, (topo_edge, face1_index, face2_index) in enumerate(edge_face_connectivity):
            # Transform the connectivity list into pure index form.
            edge_face_connectivity[edge_index] = (edge_index, face1_index, face2_index)
            
            # Type check
            curve = BRepAdaptor_Curve(topo_edge)
            if curve.GetType() not in [GeomAbs_Circle, GeomAbs_Line, GeomAbs_Ellipse, GeomAbs_BSplineCurve]:
                raise ValueError("Unsupported curve type: {}".format(curve.GetType()))
            
            # Sample points
            # Because we're gonna handling the half-edge structure, the orientation of an edge is very important
            param_start = curve.FirstParameter() if topo_edge.Orientation() == 0 else curve.LastParameter()
            param_end = curve.LastParameter() if topo_edge.Orientation() == 0 else curve.FirstParameter()
            
            params = np.linspace(param_start, param_end, num=sample_resolution)
            sample_points = []
            for u in params:
                pnt = curve.Value(u)
                first_derivative = gp_Vec()
                curve.D1(u, pnt, first_derivative) # first_derivative is the C type output parameter 
                first_derivative = first_derivative.Normalized()
                sample_points.append(np.array([pnt.X(), pnt.Y(), pnt.Z(), first_derivative.X(), first_derivative.Y(), first_derivative.Z()], dtype=np.float32))
            
            # Check the closest point’s distance to the center of all sample points,
            # If it's too small, we consider this model has some problems.
            sample_points = np.stack(sample_points, axis=0)
            edge_center = np.mean(sample_points[:, :3], axis=0)
            edge_scale = np.min(np.linalg.norm(sample_points[:, :3] - edge_center, axis=1))
            if edge_scale < 1e-3:
                raise ValueError("Edge scale too small: {}".format(edge_scale))
            edge_sample_points.append(np.stack(sample_points, axis=0))
        
        # Topology structure 
        edge_face_connectivity = np.asarray(edge_face_connectivity, dtype=np.int32)
        face_adj = np.zeros((num_faces, num_faces), dtype=bool)
        face_adj[edge_face_connectivity[:, 1], edge_face_connectivity[:, 2]] = True
        zero_positions = np.stack(np.where(face_adj == 0), axis=1) # This is for the convenience of the training, it indicates which two faces are not intersected.

        edge_sample_points = np.stack(edge_sample_points, axis=0)
        
        result = {
            'face_sample_points'        : edge_sample_points.astype(np.float32),
            'half_edge_sample_points'   : face_sample_points.astype(np.float32),
            'edge_face_connectivity'    : edge_face_connectivity.astype(np.int64),
            "face_adj_matrix"           : face_adj,
            "non_intersection_index"    : zero_positions,
        }
        
        # Finally, we've done.
        np.savez_compressed(uv_grid_path, **result)
    
    def export_photos(self, photo_path):
        renderer = PhotoRenderer(self.solid)
        svr_images, mvr_images = renderer.process()
        np.savez_compressed(photo_path, svr=svr_images, mvr=mvr_images)
    
    def _create_dir_if_not_exist(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    def _extract_edges_from_faces(self, topo_face):
        """
        Extract edges from the correspond face 
        Arg:
            - topo_face(topods.Face): Face to extract
        Return:
            - edge(List): A list of lists of edges. Every list of edges represent the edges extracted from the correspond wire. 
        """
        edges = []
        for wire in extract_primitives(topo_face, TopAbs_WIRE):
            wire_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
            local_edges = []
            while wire_explorer.More():
                edge = topods.Edge(wire_explorer.Current())
                local_edges.append(edge)
                wire_explorer.Next()
            edges.append(local_edges)
        return edges
    