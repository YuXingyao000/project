import trimesh
from model.AdaPoinTr import AdaPoinTr
from dataset.dataset import DeepCADDataset
import torch

class Inferer:
    def __init__(self, model, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        
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
                
if __name__ == "__main__":
    model = AdaPoinTr()
    inferer = Inferer(model, model_path="runs/AdaPoinTr_DeepCAD_Aug14_09-21-49/best_model.pth")
    dataset = DeepCADDataset(
        data_root='/mnt/d/data/processed_deepcad',
        index_path='/mnt/d/data/DeepCAD/data_index/deduplicated_deepcad_testing_7_30.txt'
    )
    i = 8
    gt_brep_grid, gt_point_cloud, partial_point_cloud, cropped_point_cloud = dataset[i]
    coarse_point_cloud, rebuild_points = inferer.single_infer(point_cloud=partial_point_cloud)
    trimesh.PointCloud(partial_point_cloud.cpu().numpy()).export("partial.ply")
    trimesh.PointCloud(gt_point_cloud.cpu().numpy()).export("gt.ply")
    trimesh.PointCloud(coarse_point_cloud.reshape(-1, 3).cpu().numpy()).export("coarse.ply")
    trimesh.PointCloud(rebuild_points.reshape(-1, 3).cpu().numpy()).export("rebuild.ply")
