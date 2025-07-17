import torch
import numpy as np
import os
from model import PoinTr
from dataset.dataset import ABCDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_model(checkpoint_path, device):
    # the trained model from checkpoint    
    model = PoinTr()
    model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['val_loss']:.6f}")
    print(f"Training loss: {checkpoint['train_loss']:.6f}")
    return model

def test_model(model, test_dataloader, device):
    # Test the model and return numpy arrays for evaluation
    model.eval()
    
    all_gt_point_clouds = []
    all_gt_brep_grids = []
    all_partial_point_clouds = []
    all_predicted_point_clouds = []
    all_predicted_brep_grids = []
    all_sparse_losses = []
    all_brep_losses = []
    
    print("Running inference...")
    
    with torch.no_grad():
        for i, (gt_brep_grid, gt_point_cloud, partial_point_cloud, cropped_point_cloud) in tqdm(enumerate(test_dataloader), desc="Testing"):
            gt = gt_point_cloud.to(device)
            gt_brep_grid = gt_brep_grid.to(device)
            partial = partial_point_cloud.to(device)
            
            coarse_point_cloud, rebuild_points, brep_grids = model(partial)
            sparse_loss, brep_loss = model.get_loss(coarse_point_cloud, rebuild_points, brep_grids, gt, gt_brep_grid)
            
            all_gt_point_clouds.append(gt.cpu().numpy())
            all_gt_brep_grids.append(gt_brep_grid.cpu().numpy())
            all_partial_point_clouds.append(partial.cpu().numpy())
            all_predicted_point_clouds.append(rebuild_points.cpu().numpy())
            all_predicted_brep_grids.append(brep_grids.cpu().numpy())
            all_sparse_losses.append(sparse_loss.item())
            all_brep_losses.append(brep_loss.item())
            
            if i == 0:
                import trimesh
                for i in range(gt.shape[0]):
                    os.makedirs(f'visualization/{i}', exist_ok=True)
                    trimesh.PointCloud(gt[i].cpu().numpy().reshape(-1, 3)).export(f'visualization/{i}/gt.ply')
                    trimesh.PointCloud(partial[i].cpu().numpy().reshape(-1, 3)).export(f'visualization/{i}/partial.ply')
                    trimesh.PointCloud(rebuild_points[i].cpu().numpy().reshape(-1, 3)).export(f'visualization/{i}/rebuild_points.ply')
                    trimesh.PointCloud(brep_grids[i].cpu().numpy().reshape(-1, 3)).export(f'visualization/{i}/brep_grids.ply')
    
    results = {
        'gt_point_clouds': np.concatenate(all_gt_point_clouds, axis=0),
        'gt_brep_grids': np.concatenate(all_gt_brep_grids, axis=0),
        'partial_point_clouds': np.concatenate(all_partial_point_clouds, axis=0),
        'predicted_point_clouds': np.concatenate(all_predicted_point_clouds, axis=0),
        'predicted_brep_grids': np.concatenate(all_predicted_brep_grids, axis=0),
        'sparse_losses': np.array(all_sparse_losses),
        'brep_losses': np.array(all_brep_losses),
        'total_losses': np.array(all_sparse_losses) + np.array(all_brep_losses)
    }
    
    print(f"\nTest Results Summary:")
    print(f"Number of samples: {len(results['gt_point_clouds'])}")
    print(f"Average sparse loss: {np.mean(results['sparse_losses']):.6f}")
    print(f"Average brep loss: {np.mean(results['brep_losses']):.6f}")
    print(f"Average total loss: {np.mean(results['total_losses']):.6f}")
    
    print(f"\nArray Shapes: {results['gt_point_clouds'].shape}")
    print(f"GT BREP grids: {results['gt_brep_grids'].shape}")
    print(f"Partial point clouds: {results['partial_point_clouds'].shape}")
    print(f"Predicted point clouds: {results['predicted_point_clouds'].shape}")
    print(f"Predicted BREP grids: {results['predicted_brep_grids'].shape}")
    return results

def main():
    # Configuration
    checkpoint_path = 'runs/training_distributed_Jul16_15-44-31/checkpoint_epoch_600.pth'
    data_root = '/mnt/d/data/1000_16_brep_sample_rate_processed_data'
    mode = 'test'  #orval' ortrain
    batch_size =8
    num_workers =4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = load_model(checkpoint_path, device)
    
    test_dataset = ABCDataset(root=data_root, mode=mode)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=False
    )
    
    print(f"Testing on {len(test_dataset)} samples from {mode} set")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {len(test_dataloader)}")
    
    results = test_model(model, test_dataloader, device)
    
    print("\nTesting completed!")
    return results

if __name__ == "__main__":
    results = main() 