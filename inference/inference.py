from inference.Inferer import Inferer
import h5py
import numpy as np
import torch

if __name__ == "__main__":
    inferer = Inferer(model_path="runs/training_Aug06_11-24-36/best_model.pth")
    h5_path = "/mnt/d/data/1000_16_brep_sample_rate_processed_data/00000003/00000003_cropped_pc.h5"
    with h5py.File(h5_path, 'r') as f:
        # Randomly select one of the 64 preprocessed samples
        num_samples = int(f.attrs['num_samples'])
        sample_idx = np.random.randint(0, num_samples)
        # Load the selected sample and convert to numpy array
        partial_data = f['input_data'][sample_idx]
        crop_data = f['crop_data'][sample_idx]
        partial_point_cloud = torch.from_numpy(np.array(partial_data)).float()  # (2048, 3)
        cropped_point_cloud = torch.from_numpy(np.array(crop_data)).float()   # (2048, 3)
        inferer.single_infer(point_cloud=partial_point_cloud)