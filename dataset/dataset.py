import os
import torch
import numpy as np
import trimesh
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

import transform as transform 

class ABCDataset(Dataset):
    def __init__(self, root, level='simple', mode='train'):
        self.root = Path(root)
        
        self.level = level
        assert self.level in ['simple', 'moderate', 'hard'], f"Invalid level: {self.level}"
        
        self.mode = mode
        assert self.mode in ['train', 'val', 'test'], f"Invalid mode: {self.mode}"
        if self.mode == 'train':
            self.model_ids = os.listdir(self.root)[:int(len(os.listdir(self.root)) * 0.75)]
        elif self.mode == 'val':
            self.model_ids = os.listdir(self.root)[int(len(os.listdir(self.root)) * 0.75) : int(len(os.listdir(self.root)) * 0.875)]
        elif self.mode == 'test':
            self.model_ids = os.listdir(self.root)[int(len(os.listdir(self.root)) * 0.875):]
        
        
    def __len__(self):
        return len(self.model_ids)
    
    def __getitem__(self, idx):
        data = {}
        model_id = self.model_ids[idx]
        
        # GT Brep Grid (output)
        gt_brep_grid = np.load(self.root / model_id / f"{model_id}.npz")["face_sample_points"] # (face_num, sample_num, sample_num, 6)
        data["gt_brep_grid"] = gt_brep_grid
        
        # Drop normal vectors, keep only position vectors (first 3 dimensions)
        gt_brep_grid = gt_brep_grid[:, :, :, :3]  # (face_num, sample_num, sample_num, 3)
        
        # Pad BREP grid to fixed size (30, 32, 32, 3)
        num_faces = gt_brep_grid.shape[0]
        max_faces = 30
        
        if num_faces < max_faces:
            # Pad with zeros for missing faces
            padding_shape = (max_faces - num_faces, 32, 32, 3)
            padding = np.zeros(padding_shape, dtype=gt_brep_grid.dtype)
            gt_brep_grid = np.concatenate([gt_brep_grid, padding], axis=0)
        elif num_faces > max_faces:
            # Truncate if more than max faces (shouldn't happen based on your description)
            gt_brep_grid = gt_brep_grid[:max_faces]
        
        # GT Point Cloud (Maybe useful for training)
        gt_point_cloud = trimesh.load(self.root / model_id / f"{model_id}.ply")
        gt_point_cloud = np.array(gt_point_cloud.vertices)  # type: ignore # (8192, 3)
        data["gt_point_cloud"] = gt_point_cloud  # type: ignore

        # Convert to tensor first
        gt_point_cloud_tensor = torch.from_numpy(data["gt_point_cloud"]).float()
        
        assert gt_point_cloud_tensor.shape[0] == 8192, f"Point cloud has {gt_point_cloud_tensor.shape[0]} points, expected 8192"
        assert isinstance(gt_point_cloud_tensor, torch.Tensor), "Point cloud must be a torch tensor"
        
        # Generate partial and cropped point clouds for training
        if self.mode == 'train':
            # Add batch dimension for SeparatePointCloud
            gt_point_cloud_batch = gt_point_cloud_tensor.unsqueeze(0)  # (1, 8192, 3)
            
            # Create separate transform for partial/crop generation
            separate_transform = transform.SeparatePointCloud(
                num_points=8192,
                padding_zeros=False
            )
            
            # Generate partial and cropped data
            partial_point_cloud, cropped_point_cloud = separate_transform(
                gt_point_cloud_batch, 
                crop=[2048, 6144]  # Random crop between 25% and 75%
            )
            
            # Remove batch dimension
            partial_point_cloud = partial_point_cloud.squeeze(0)  # (N, 3) where N varies
            cropped_point_cloud = cropped_point_cloud.squeeze(0)  # (M, 3) where M varies
            
            # Downsample partial point cloud to 2048 points
            if partial_point_cloud.shape[0] > 2048:
                # Randomly sample if too many points
                choice = torch.randperm(partial_point_cloud.shape[0])
                partial_point_cloud = partial_point_cloud[choice[:2048]]
            elif partial_point_cloud.shape[0] < 2048:
                # Pad with zeros if too few points
                zeros = torch.zeros(2048 - partial_point_cloud.shape[0], 3)
                partial_point_cloud = torch.cat([partial_point_cloud, zeros], dim=0)
            
            # Downsample cropped point cloud to 2048 points
            if cropped_point_cloud.shape[0] > 2048:
                # Randomly sample if too many points
                choice = torch.randperm(cropped_point_cloud.shape[0])
                cropped_point_cloud = cropped_point_cloud[choice[:2048]]
            elif cropped_point_cloud.shape[0] < 2048:
                # Pad with zeros if too few points
                zeros = torch.zeros(2048 - cropped_point_cloud.shape[0], 3)
                cropped_point_cloud = torch.cat([cropped_point_cloud, zeros], dim=0)
        else:
            # For validation/test, just use the complete point cloud
            partial_point_cloud = gt_point_cloud_tensor
            cropped_point_cloud = torch.zeros(8192, 3)  # Empty cropped data
        
        return torch.from_numpy(gt_brep_grid).float(), gt_point_cloud_tensor, partial_point_cloud, cropped_point_cloud
        
    
if __name__ == "__main__":
    dataset = ABCDataset(root='/mnt/d/data/processed_data_step', mode='train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    for idx, (gt_brep_grid, gt_point_cloud, partial_point_cloud, cropped_point_cloud) in enumerate(dataloader):
        print(gt_brep_grid.shape)
        print(gt_point_cloud.shape)
        print(partial_point_cloud.shape)
        print(cropped_point_cloud.shape)
        break