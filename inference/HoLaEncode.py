import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from model.HoLa.HoLaAutoEncoder import HoLaAutoEncoder

class EncodeDataset(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = Path(data_root)
        self.data_folders = sorted([self.data_root / folder_id for folder_id in os.listdir(self.data_root)])
    
    def __len__(self):
        return len(self.data_folders)
    
    def __getitem__(self, index):
        raw_uv_data = np.load(self.data_folders[index] / 'uv_grids.npz')
        return (
            self.data_folders[index].name,
            raw_uv_data['face_sample_points'],
            raw_uv_data['half_edge_sample_points'],
            raw_uv_data['edge_face_connectivity'],
            raw_uv_data["non_intersection_index"]
        )
    
    @staticmethod
    def collate_fn(batch):
        # Unzip the batch into 5 lists
        data_ids, face_sample_points, half_edge_sample_points, edge_face_connectivity, non_intersection_index = zip(*batch)
        
        # Record how many faces and half-edges a model has
        num_faces  = torch.tensor([faces_per_model.shape[0] for faces_per_model in face_sample_points], dtype=torch.int32)
        accum_num_faces = torch.cat([torch.zeros(1, dtype=torch.int32), num_faces.cumsum(0)])
        
        num_edges  = torch.tensor([edges_per_model.shape[0] for edges_per_model in half_edge_sample_points], dtype=torch.int32)
        accum_num_edges = torch.cat([torch.zeros(1, dtype=torch.int32), num_edges.cumsum(0)])

        # Convert to tensors if they aren’t already
        edge_face_connectivity = [torch.as_tensor(conn) for conn in edge_face_connectivity]
        non_intersection_index = [torch.as_tensor(idx) for idx in non_intersection_index]

        # Repeat offsets to match each row
        edge_offsets = torch.repeat_interleave(accum_num_edges[:-1], num_edges)
        face_offsets = torch.repeat_interleave(accum_num_faces[:-1], num_edges)

        # Flatten and shift in one go
        edge_face_connectivity = torch.cat(edge_face_connectivity, dim=0)
        edge_face_connectivity[:, 0] += edge_offsets
        edge_face_connectivity[:, 1:] += face_offsets[:, None]

        # ---- attn_mask: (total_faces, total_faces) ----
        batch_size = len(data_ids)
        model_ids = torch.repeat_interleave(torch.arange(batch_size), num_faces)
        same_model = model_ids.unsqueeze(0) == model_ids.unsqueeze(1) # broadcast [1, total_num_faces] -> [total_num_faces, total_num_faces]; [total_num_faces, 1] -> [total_num_faces, total_num_faces] 
        attn_mask = ~same_model  # True = block, False = allow
        
        face_sample_points = [torch.as_tensor(f) for f in face_sample_points]
        all_faces = torch.cat(face_sample_points, dim=0)
        
        half_edge_sample_points = [torch.as_tensor(f) for f in half_edge_sample_points]
        all_half_edges = torch.cat(half_edge_sample_points, dim=0)
        
        return (
            data_ids,
            all_faces,
            all_half_edges, 
            attn_mask, 
            edge_face_connectivity,
            num_faces
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to the data root")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to the model file(.ckpt)")
    
    args = parser.parse_args()
    data_root = Path(args.data_root)
    model_file = Path(args.model_ckpt)
    
    checkpoint = torch.load(model_file, weights_only=False)["state_dict"]
    model = HoLaAutoEncoder()
    model.load_state_dict(checkpoint)
    
    deepcad_dataset = EncodeDataset(data_root)
    encode_dataloader = DataLoader(deepcad_dataset, 
                                   batch_size=128, 
                                   shuffle=False, 
                                   num_workers=2, 
                                   collate_fn=EncodeDataset.collate_fn, 
                                   pin_memory=True, 
                                   persistent_workers=True,
                                   drop_last=False)
    
    pbar = tqdm(encode_dataloader, desc=f"Encoding...", leave=False)
    for i, batch in enumerate(pbar):
        data_id, all_faces_sample_points, all_half_edges_sample_points, face_attn_mask, edge_face_connectivity, num_faces_per_model = batch
        face_latent_features, half_edge_features = model.encode(all_faces_sample_points, all_half_edges_sample_points, face_attn_mask, edge_face_connectivity, num_faces_per_model)
        _, mean, std = model.sample(face_latent_features)
        pass