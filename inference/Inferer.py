import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model.PoinTr import PoinTr

class Inferer:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PoinTr()
        
        # Load the checkpoint dictionary
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Check if it's a full checkpoint or just state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # It's a full checkpoint with multiple keys
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # It's just the state dict
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        self.model.to(self.device)

    def single_infer(self, point_cloud: torch.Tensor):
        with torch.no_grad():
            point_cloud = point_cloud.to(self.device).unsqueeze(0)
            coarse_point_cloud, rebuild_points = self.model(point_cloud)
            return coarse_point_cloud, rebuild_points
    
    def batch_infer(self, pc_dataset: Dataset, batch_size=16):
        dataloader = DataLoader(pc_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
        with torch.no_grad():
            for partial_point_cloud in tqdm(dataloader, desc="Inferring", leave=False):
                # Clear cache before processing each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                partial_point_cloud.to(self.device)
                coarse_point_cloud, rebuild_points = self.model(partial_point_cloud)

