import os
import torch
import numpy as np
import trimesh
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

import dataset.transform as transform 

class ABCDataset(Dataset):
    def __init__(self, root, level='simple', mode='train'):
        self.root = Path(root)
        
        self.level = level
        assert self.level in ['simple', 'moderate', 'hard'], f"Invalid level: {self.level}"
        
        self.mode = mode
        assert self.mode in ['train', 'val', 'test'], f"Invalid mode: {self.mode}"
        
        all_ids = [model_id for model_id in os.listdir(self.root) if os.path.isdir(self.root / model_id)]
        
        # Get model IDs based on mode
        if self.mode == 'train':
            self.model_ids = all_ids[:int(len(all_ids) * 0.75)]
        elif self.mode == 'val':
            self.model_ids = all_ids[int(len(all_ids) * 0.75) : int(len(all_ids) * 0.875)]
        elif self.mode == 'test':
            self.model_ids = all_ids[int(len(all_ids) * 0.875):]
        
        # Filter for models that have all required files
        self.model_ids = [
            model_id for model_id in self.model_ids 
            if (self.root / model_id / f"{model_id}.npz").exists() and  # BREP grids
               (self.root / model_id / f"{model_id}_pc.npz").exists() and  # Full point cloud
               (self.root / model_id / f"{model_id}_cropped_pc.h5").exists()  # Preprocessed cropped data
        ]
        
        print(f"Found {len(self.model_ids)} models for {self.mode} mode")
        
    def __len__(self):
        return len(self.model_ids)
    
    def __getitem__(self, idx):
        model_id = self.model_ids[idx]
        
        # Load BREP grids (shape: [num_faces, 16, 16, 3])
        brep_data = np.load(self.root / model_id / f"{model_id}.npz")
        gt_brep_grid = brep_data["face_sample_points"]  # (face_num, 16, 16, 6)
        
        # Drop normal vectors, keep only position vectors (first 3 dimensions)
        gt_brep_grid = gt_brep_grid[:, :, :, :3]  # (face_num, 16, 16, 3)
        
        # Pad BREP grid to fixed size (30, 16, 16, 3)
        num_faces = gt_brep_grid.shape[0]
        max_faces = 30
        
        if num_faces < max_faces:
            # Pad with zeros for missing faces
            padding_shape = (max_faces - num_faces, 16, 16, 3)
            padding = np.zeros(padding_shape, dtype=gt_brep_grid.dtype)
            gt_brep_grid = np.concatenate([gt_brep_grid, padding], axis=0)
        elif num_faces > max_faces:
            # Truncate if more than max faces
            gt_brep_grid = gt_brep_grid[:max_faces]
        
        # Load full point cloud (shape: [8192, 3])
        pc_data = np.load(self.root / model_id / f"{model_id}_pc.npz")
        gt_point_cloud = pc_data["points"]  # (8192, 3)
        
        # Convert to tensors
        gt_brep_grid_tensor = torch.from_numpy(gt_brep_grid).float()
        gt_point_cloud_tensor = torch.from_numpy(gt_point_cloud).float()
        
        # Validate shapes
        assert gt_point_cloud_tensor.shape == (8192, 3), f"Point cloud has shape {gt_point_cloud_tensor.shape}, expected (8192, 3)"
        assert gt_brep_grid_tensor.shape == (30, 16, 16, 3), f"BREP grid has shape {gt_brep_grid_tensor.shape}, expected (30, 16, 16, 3)"
        
        # Load preprocessed cropped data
        if self.mode == 'train':
            # Load from HDF5 file and randomly select one sample
            import h5py
            with h5py.File(self.root / model_id / f"{model_id}_cropped_pc.h5", 'r') as f:
                # Randomly select one of the 64 preprocessed samples
                num_samples = int(f.attrs['num_samples'])
                sample_idx = np.random.randint(0, num_samples)
                # Load the selected sample and convert to numpy array
                partial_data = f['input_data'][sample_idx]
                crop_data = f['crop_data'][sample_idx]
                partial_point_cloud = torch.from_numpy(np.array(partial_data)).float()  # (2048, 3)
                cropped_point_cloud = torch.from_numpy(np.array(crop_data)).float()   # (2048, 3)
        else:
            # For validation/test, use the full point cloud as partial and zeros as cropped
            partial_point_cloud = gt_point_cloud_tensor
            cropped_point_cloud = torch.zeros(2048, 3)  # Empty cropped data
        
        return gt_brep_grid_tensor, gt_point_cloud_tensor, partial_point_cloud, cropped_point_cloud
        
    
if __name__ == "__main__":
    dataset = ABCDataset(root='/mnt/d/data/processed_data_step', mode='train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    for idx, (gt_brep_grid, gt_point_cloud, partial_point_cloud, cropped_point_cloud) in enumerate(dataloader):
        print(gt_brep_grid.shape)
        print(gt_point_cloud.shape)
        print(partial_point_cloud.shape)
        print(cropped_point_cloud.shape)
        break