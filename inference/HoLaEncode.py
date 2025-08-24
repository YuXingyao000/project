import os
import argparse
import torch
import warnings
import numpy as np
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.console import Console
from torch.utils.data import Dataset, DataLoader

from model.HoLa.HoLaAutoEncoder import HoLaAutoEncoder

# Filter some ugly messages.
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False")

class EncodeDataset(Dataset):
    """ 
    ## If you want to use your own data, please be sure you have the following data:
    1. face sample points(UV) for each model
    2. half-edge sample points(U) for each model
    3. edge-face connectivity, a list of triphlets with the elements indicating [edge_index, face1_index, face2_index] for each model.
    """
    def __init__(self, data_root: str):
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
        )
    

    @staticmethod
    def collate_fn(batch):
        """
        Theoratically, you don't need to change this function.
        But please be sure you didn't pad face sample points and half-edge sample points.
        Normally, they should be a list containing the sampling points of different lengths for each model.
        """
        # Unzip the batch
        data_ids, face_sample_points, half_edge_sample_points, edge_face_connectivity = zip(*batch)
        
        # Number of faces/edges per model
        num_faces  = torch.tensor([f.shape[0] for f in face_sample_points], dtype=torch.int32)
        accum_num_faces = torch.cat([torch.zeros(1, dtype=torch.int32), num_faces.cumsum(0)])
        
        num_edges  = torch.tensor([e.shape[0] for e in half_edge_sample_points], dtype=torch.int32)
        accum_num_edges = torch.cat([torch.zeros(1, dtype=torch.int32), num_edges.cumsum(0)])

        # Flatten connectivity with offsets
        edge_face_connectivity = [torch.as_tensor(conn) for conn in edge_face_connectivity]
        edge_offsets = torch.repeat_interleave(accum_num_edges[:-1], num_edges)
        face_offsets = torch.repeat_interleave(accum_num_faces[:-1], num_edges)
        
        edge_face_connectivity = torch.cat(edge_face_connectivity, dim=0)
        edge_face_connectivity[:, 0] += edge_offsets
        edge_face_connectivity[:, 1:] += face_offsets[:, None]

        # Attention mask
        batch_size = len(data_ids)
        model_ids = torch.repeat_interleave(torch.arange(batch_size), num_faces)
        same_model = model_ids.unsqueeze(0) == model_ids.unsqueeze(1)
        attn_mask = ~same_model

        # Flatten faces and half-edges
        all_faces = torch.cat([torch.as_tensor(f) for f in face_sample_points], dim=0)
        all_half_edges = torch.cat([torch.as_tensor(f) for f in half_edge_sample_points], dim=0)
        
        return (
            data_ids,
            all_faces,
            all_half_edges, 
            attn_mask, 
            edge_face_connectivity,
            num_faces
        )


def encode_and_save(data_root: Path, model_file: Path, batch_size: int = 128, num_workers: int = 2):
    """Encodes all data with the model and saves per-model latent mean/std."""

    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Stage 1: Load model ----------------
    with console.status("[cyan]Loading model...") as status:
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)["state_dict"]
        model = HoLaAutoEncoder().to(device)
        model.load_state_dict(checkpoint)
        model.eval()
    console.log("[green]✅ Model loaded successfully")
    
    # ---------------- Stage 2: Prepare dataset ----------------
    with console.status("[cyan]Preparing dataset...") as status:
        dataset = EncodeDataset(data_root)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=EncodeDataset.collate_fn,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )
    console.log(f"[green]✅ Dataset prepared: {len(dataset)} models found")

    # ---------------- Stage 3: Encode ----------------
    total_faces_processed = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} models processed"),
        console=console,
        transient=True  # progress bar disappears when done
    ) as progress:
        task = progress.add_task("Encoding models...", total=len(dataset))

        for data_ids, all_faces, all_half_edges, attn_mask, edge_face_connectivity, num_faces_per_model in dataloader:
            all_faces = all_faces.to(device)
            all_half_edges = all_half_edges.to(device)
            attn_mask = attn_mask.to(device)
            edge_face_connectivity = edge_face_connectivity.to(device)
            num_faces_per_model = num_faces_per_model.to(device)

            with torch.no_grad():
                face_latent_features, _ = model.encode(
                    all_faces, all_half_edges, attn_mask, edge_face_connectivity, num_faces_per_model
                )
                _, mean, std = model.sample(face_latent_features)  # [total_faces, 32]

            # Split back per-model
            split_means = torch.split(mean.cpu(), num_faces_per_model.tolist(), dim=0)
            split_stds  = torch.split(std.cpu(),  num_faces_per_model.tolist(), dim=0)

            # Safety check
            assert len(split_means) == len(split_stds) == len(num_faces_per_model), \
                "Mismatch in number of models when splitting mean and std"
            for i, (m, s, n) in enumerate(zip(split_means, split_stds, num_faces_per_model.tolist())):
                assert m.shape[0] == s.shape[0] == n, \
                    f"Mismatch at model {i}: mean={m.shape[0]}, std={s.shape[0]}, expected={n}"

            # Save per model
            for folder_name, m, s in zip(data_ids, split_means, split_stds):
                out_file = Path(data_root) / folder_name / "latents.npz"
                np.savez_compressed(out_file, mean=m.numpy(), std=s.numpy())

            total_faces_processed += len(data_ids)
            progress.update(task, advance=len(data_ids))

    console.log(f"[bold green]✅ All {len(dataset)} models encoded successfully! Total faces: {total_faces_processed}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to the data root")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to the model file (.ckpt)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    encode_and_save(
        data_root=Path(args.data_root),
        model_file=Path(args.model_ckpt),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()