import torch
import os.path
import numpy as np
import open3d as o3d

from pathlib import Path


def normalize_coord(point_cloud_with_normals):
    """
    Normalize point clouds with normals to unit scale and zero center.
    
    Args:
        point_cloud_with_normals: Tensor [..., 6] where first 3 dims are XYZ coordinates,
                                 last 3 dims are normal vectors
    
    Returns:
        normalized_points: Points normalized to [-1, 1] range
        normalized_normals: Unit normals (preserved direction after scaling)
        center: Original center point for each object [num_objects, 3]
        scale: Original scale factor for each object [num_objects, 1]
    """
    # Split coordinates and normals
    xyz_coords = point_cloud_with_normals[..., :3]  # [N, num_points, 3]
    normal_vectors = point_cloud_with_normals[..., 3:]  # [N, num_points, 3]
    
    original_shape = xyz_coords.shape
    num_objects = original_shape[0]
    
    # Reshape for batch processing: [num_objects, total_points, 3]
    xyz_coords = xyz_coords.reshape(num_objects, -1, 3)
    normal_vectors = normal_vectors.reshape(num_objects, -1, 3)
    
    # Calculate "target points" (where normals point to)
    # This helps preserve normal direction during scaling
    target_points = xyz_coords + normal_vectors
    
    # Find center and scale for normalization
    center_point = xyz_coords.mean(dim=1, keepdim=True)  # [N, 1, 3]
    distances_from_center = torch.linalg.norm(xyz_coords - center_point, dim=-1)  # [N, total_points]
    max_distance = distances_from_center.max(dim=1, keepdims=True)[0]  # [N, 1]
    
    # Ensure we don't divide by zero
    assert max_distance.min() > 1e-3, "Object too small - max distance from center < 1e-3"
    
    # Normalize coordinates to [-1, 1] range
    normalized_coords = (xyz_coords - center_point) / (max_distance[:, None] + 1e-6)
    normalized_targets = (target_points - center_point) / (max_distance[:, None] + 1e-6)
    
    # Recalculate normals after scaling (maintains direction)
    normalized_normals = normalized_targets - normalized_coords
    # Ensure unit length
    normalized_normals = normalized_normals / (1e-6 + torch.linalg.norm(normalized_normals, dim=-1, keepdim=True))
    
    # Reshape back to original dimensions
    normalized_coords = normalized_coords.reshape(original_shape)
    normalized_normals = normalized_normals.reshape(original_shape)
    
    return normalized_coords, normalized_normals, center_point[:, 0], max_distance



def prepare_condition(v_cond_root, v_folder_path):
    condition = {
    }
    pc = o3d.io.read_point_cloud(str(v_cond_root / v_folder_path / "pc.ply"))
    points = np.concatenate((np.asarray(pc.points), np.asarray(pc.normals)), axis=-1)
    condition["points"] = torch.from_numpy(points).float()[None,]
    return condition
        

# This is just for inference, not used for training.
class HoLaAutoEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, model_ids_file):
        super(HoLaAutoEncoderDataset, self).__init__()

        self.data_folders = [item.strip() for item in open(model_ids_file).readlines()]
        self.root = Path(data_root)
        self.data_folders = [folder for folder in self.data_folders if os.path.exists(self.root / folder / "data.npz")]

        print(f"{len(self.data_folders)} models loaded")

    def __len__(self):
        return len(self.data_folders)

    def __getitem__(self, idx):
        # Read the data
        prefix = self.data_folders[idx]
        data_npz = np.load(str(self.root / prefix / "data.npz"))
        
        # Raw faces' and edges' uv sample points. The resolution is 16.
        face_points = torch.from_numpy(data_npz['sample_points_faces'])
        edge_points = torch.from_numpy(data_npz['sample_points_lines'])

        # Raw face adjacency matrix.
        face_adj = torch.from_numpy(data_npz['face_adj'])
        # Raw edge-face connectivity. [edge_num, face1_num, face2_num]
        edge_face_connectivity = torch.from_numpy(data_npz['edge_face_connectivity'])
        # Remove the edges that are generated from the face's self-intersection.
        edge_face_connectivity = edge_face_connectivity[edge_face_connectivity[:, 1] != edge_face_connectivity[:, 2]]

        # Non-intersection face pair. Indicate which two faces do not intersect.
        zero_positions = torch.from_numpy(data_npz['zero_positions'])
        if zero_positions.shape[0] > edge_face_connectivity.shape[0]:
            index = np.random.choice(zero_positions.shape[0], edge_face_connectivity.shape[0], replace=False)
            zero_positions = zero_positions[index]

        # Normalize the face and edge points.
        # This will normalize and save the bounding box(scale and position) of every SINGLE face
        face_points_norm, face_normal_norm, face_center, face_scale = normalize_coord(face_points)
        # The same as the above
        edge_points_norm, edge_normal_norm, edge_center, edge_scale = normalize_coord(edge_points)

        # For loss computing
        face_norm = torch.cat((face_points_norm, face_normal_norm), dim=-1)
        edge_norm = torch.cat((edge_points_norm, edge_normal_norm), dim=-1)

        # For loss computing
        face_bbox = torch.cat((face_center, face_scale), dim=-1)
        edge_bbox = torch.cat((edge_center, edge_scale), dim=-1)

        # Point cloud
        condition = prepare_condition(self.root, prefix)

        return (
            prefix,
            face_points, edge_points,
            face_norm, edge_norm,
            face_bbox, edge_bbox,
            edge_face_connectivity, zero_positions, face_adj,
            condition
        )

    @staticmethod
    def collate_fn(batch):
        # Some additional information.
        (
            prefix,
            face_points, edge_points,
            face_norm, edge_norm,
            face_bbox, edge_bbox,
            edge_face_connectivity, zero_positions, face_adj,
            conditions
        ) = zip(*batch)
        bs = len(prefix)

        # Flatten the face indices. 
        # The number of faces of each model are not equal, we need to flatten them so the PyTorch can handle them.
        flat_zero_positions = []
        num_face_record = []

        num_faces = 0
        num_edges = 0
        edge_conn_num = []
        for i in range(bs):
            edge_face_connectivity[i][:, 0] += num_edges
            edge_face_connectivity[i][:, 1:] += num_faces
            edge_conn_num.append(edge_face_connectivity[i].shape[0])
            flat_zero_positions.append(zero_positions[i] + num_faces)
            num_faces += face_norm[i].shape[0]
            num_edges += edge_norm[i].shape[0]
            num_face_record.append(face_norm[i].shape[0])
        flat_zero_positions = torch.cat(flat_zero_positions, dim=0)
        
        num_face_record = torch.tensor(num_face_record, dtype=torch.long)
        num_sum_edges = sum(edge_conn_num)
        
        # Edge attention mask. During the training, each edge should only be able to attend to the edges from its own model
        edge_attn_mask = torch.ones((num_sum_edges, num_sum_edges), dtype=bool)
        id_cur = 0
        for i in range(bs):
            edge_attn_mask[id_cur:id_cur + edge_conn_num[i], id_cur:id_cur + edge_conn_num[i]] = False
            id_cur += edge_conn_num[i]

        # Face attention mask. During the training, each face should only be able to attend to the faces from its own model
        num_max_faces = num_face_record.max()
        valid_mask = torch.zeros((bs, num_max_faces), dtype=bool)
        for i in range(bs):
            valid_mask[i, :num_face_record[i]] = True
        attn_mask = torch.ones((num_faces, num_faces), dtype=bool)
        id_cur = 0
        for i in range(bs):
            attn_mask[id_cur:id_cur + face_norm[i].shape[0], id_cur: id_cur + face_norm[i].shape[0]] = False
            id_cur += face_norm[i].shape[0]

        # Prepare the condition.
        keys = conditions[0].keys()
        condition_out = {key: [] for key in keys}
        for idx in range(len(conditions)):
            for key in keys:
                condition_out[key].append(conditions[idx][key])

        for key in keys:
            condition_out[key] = torch.stack(condition_out[key], dim=0) if isinstance(condition_out[key][0], torch.Tensor) else condition_out[key]

        return {
            "v_prefix"              : prefix,
            "face_points"             : torch.cat(face_points, dim=0).to(torch.float32),
            "face_norm"             : torch.cat(face_norm, dim=0).to(torch.float32),
            "edge_points"             : torch.cat(edge_points, dim=0).to(torch.float32),
            "edge_norm"             : torch.cat(edge_norm, dim=0).to(torch.float32),
            "face_bbox"             : torch.cat(face_bbox, dim=0).to(torch.float32),
            "edge_bbox"             : torch.cat(edge_bbox, dim=0).to(torch.float32),

            "edge_face_connectivity": torch.cat(edge_face_connectivity, dim=0),
            "zero_positions"        : flat_zero_positions,
            "attn_mask"             : attn_mask,
            "edge_attn_mask"        : edge_attn_mask,

            "num_face_record"       : num_face_record,
            "valid_mask"            : valid_mask,
            "conditions"   : condition_out
        }
