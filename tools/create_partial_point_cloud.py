import os
import numpy as np
import argparse
from pathlib import Path
import ray

from tools.data.PartialPointCloudSampler import PartialPointCloudSampler

def process_data_id(batch_ids, data_root, output_root, viewpoints, is_evaluation):
    """Process a batch of data_ids with ray remote function"""
    for data_id in batch_ids:
        # Search for ply file
        data_path = data_root / data_id / f"{data_id}.ply"
        if not data_path.exists():
            continue
    
        # Create output directory
        if not (output_root / data_id).exists():
            os.makedirs(output_root / data_id)

        # Sample partial point clouds
        sampler = PartialPointCloudSampler(data_path, is_evaluation=is_evaluation)
        for i, viewpoint in enumerate(viewpoints):
            partial_point_clouds = []
            for n_points in [2048, 4096, 6144]:
                sampled_point_cloud = sampler.sample_partial_point_cloud(viewpoint, n_points=n_points)
                partial_point_clouds.append(sampled_point_cloud)

            np.savez(output_root / data_id / f"{data_id}_{i}.npz", 
                     simple=partial_point_clouds[0], 
                     moderate=partial_point_clouds[1], 
                     hard=partial_point_clouds[2])
    
    return f"Processed {data_id}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--is_evaluation", action="store_true")
    parser.add_argument("--radius", type=float, default=2)
    parser.add_argument("--elevation", type=float, default=45)
    parser.add_argument("--num_azimuths", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_ray", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    is_evaluation = args.is_evaluation
    batch_size = args.batch_size

    viewpoints = PartialPointCloudSampler.generate_viewpoints(radius=args.radius, elevation=args.elevation, num_azimuths=args.num_azimuths)
    
    # Get all data IDs
    data_ids = [data_id for data_id in os.listdir(data_root) if os.path.isdir(data_root / data_id)]
    
    # Initialize Ray
    if args.use_ray:
        ray.init()
        process_data_remote = ray.remote(process_data_id)
        # Process in parallel using Ray
        futures = []
        for i in range(0, len(data_ids), batch_size):
            batch_ids = data_ids[i:min(len(data_ids), i + batch_size)]
            future = process_data_remote.remote(batch_ids, data_root, output_root, viewpoints, is_evaluation)
            futures.append(future)

        # Wait for all tasks to complete and collect results
        results = ray.get(futures)

        # Shutdown Ray
        ray.shutdown()
    else:
        for data_id in data_ids:
            # Search for ply file
            data_path = data_root / data_id / f"{data_id}.ply"
            if not data_path.exists():
                continue
            
            # Create output directory
            if not (output_root / data_id).exists():
                os.makedirs(output_root / data_id)

            # Sample partial point clouds
            sampler = PartialPointCloudSampler(data_path, is_evaluation=is_evaluation)
            for i, viewpoint in enumerate(viewpoints):
                partial_point_clouds = []
                for n_points in [2048, 4096, 6144]:
                    sampled_point_cloud = sampler.sample_partial_point_cloud(viewpoint, n_points=n_points)
                    partial_point_clouds.append(sampled_point_cloud)

                np.savez(output_root / data_id / f"{data_id}_{i}.npz", 
                         simple=partial_point_clouds[0], 
                         moderate=partial_point_clouds[1], 
                         hard=partial_point_clouds[2])

                