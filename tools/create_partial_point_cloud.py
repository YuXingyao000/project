import os
import sys
import numpy as np
import argparse
from pathlib import Path
import ray
import open3d as o3d

# Add the project root to Python path for Ray workers
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def process_data_id(batch_ids, data_root, output_root, viewpoints, is_evaluation):
    """Process a batch of data_ids with ray remote function"""
    # Import inside function to avoid Ray serialization issues
    from tools.data.PartialPointCloudSampler import PartialPointCloudSampler
    
    for data_id in batch_ids:
        # Search for ply file
        data_path = data_root / data_id / f"{data_id}.ply"
        if not data_path.exists():
            continue
    
        # Create output directory
        for n_points in [2048, 4096, 6144]:
            criterion = "simple" if n_points == 2048 else "moderate" if n_points == 4096 else "hard"
            if not (output_root / criterion / data_id).exists():
                os.makedirs(output_root / criterion / data_id)

        # Sample partial point clouds
        sampler = PartialPointCloudSampler(data_path, is_evaluation=is_evaluation)
        for n_points in [2048, 4096, 6144]:
            criterion = "simple" if n_points == 2048 else "moderate" if n_points == 4096 else "hard"
            for i, viewpoint in enumerate(viewpoints):
                sampled_point_cloud = sampler.sample_partial_point_cloud(viewpoint, n_points=n_points)
                np.savez(
                    output_root / criterion / data_id / f"{data_id}_{i}.npz",
                    pc=sampled_point_cloud,
                )
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(sampled_point_cloud)
                o3d.io.write_point_cloud(output_root / criterion / data_id / f"{data_id}_{i}.ply", pc)
            
    
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

    # Import for viewpoint generation
    from tools.data.PartialPointCloudSampler import PartialPointCloudSampler
    
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
        # Import for non-Ray execution
        from tools.data.PartialPointCloudSampler import PartialPointCloudSampler
        
        for data_id in data_ids:
            # Search for ply file
            data_path = data_root / data_id / f"{data_id}.ply"
            if not data_path.exists():
                continue
            
            # Create output directory
            for n_points in [2048, 4096, 6144]:
                criterion = "simple" if n_points == 2048 else "moderate" if n_points == 4096 else "hard"
                if not (output_root / criterion / data_id).exists():
                    os.makedirs(output_root / criterion / data_id)

            # Sample partial point clouds
            sampler = PartialPointCloudSampler(data_path, is_evaluation=is_evaluation)
            for n_points in [2048, 4096, 6144]:
                criterion = "simple" if n_points == 2048 else "moderate" if n_points == 4096 else "hard"
                for i, viewpoint in enumerate(viewpoints):
                    sampled_point_cloud = sampler.sample_partial_point_cloud(viewpoint, n_points=n_points)
                    np.savez(
                        output_root / criterion / data_id / f"{data_id}_{i}.npz",
                        pc=sampled_point_cloud,
                    )
                    pc = o3d.geometry.PointCloud()
                    pc.points = o3d.utility.Vector3dVector(sampled_point_cloud)
                    o3d.io.write_point_cloud(output_root / criterion / data_id / f"{data_id}_{i}.ply", pc)


                