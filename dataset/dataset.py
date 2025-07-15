import os
import torch
import numpy as np
import trimesh
from pathlib import Path
from torch.utils.data import Dataset

import dataset.transform as transform 

class ABCDataset(Dataset):
    def __init__(self, root, level='simple', mode='train'):
        self.root = Path(root)
        self.data_ids = os.listdir(self.root / "gt")
        self.gt_folder = self.root / "gt"
        self.partial_folder = self.root / "partial"
        self.model_ids = os.listdir(self.gt_folder)
        
        self.level = level
        assert self.level in ['simple', 'moderate', 'hard'], f"Invalid level: {self.level}"
        
        self.mode = mode
        assert self.mode in ['train', 'val', 'test'], f"Invalid mode: {self.mode}"
        
        self.transform = self.get_transform()
        
    def get_transform(self):
        if self.mode == 'train':
            return transform.Compose([
                transform.Partial(),
                transform.Downsample(),
                transform.ToTensor(),
            ])
        else:
            return transform.Compose([
                transform.ToTensor(),
            ])
        
    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, idx):
        data = {}
        model_id = self.model_ids[idx]
        
        # GT Brep Grid (output)
        gt_brep_grid = np.load(self.gt_folder / model_id / f"{model_id}.npz")["face_sample_points"] # (face_num, sample_num, sample_num, 3)
        data["gt_brep_grid"] = gt_brep_grid
        
        # GT Point Cloud (Maybe useful for training)
        gt_point_cloud = trimesh.load(self.gt_folder / model_id / f"{model_id}.ply")
        gt_point_cloud = np.array(gt_point_cloud.vertices)  # type: ignore # (8192, 3)
        data["gt_point_cloud"] = gt_point_cloud  # type: ignore

        # Randomly select a viewpoint and randomly sample a partial point cloud
        if self.transform is not None:
            data["partial_point_cloud"] = self.transform(data["gt_point_cloud"])
        else:
            data["partial_point_cloud"] = data["gt_point_cloud"]
        
        return data['gt_brep_grid'], data['gt_point_cloud'], data['partial_point_cloud']
        
    
