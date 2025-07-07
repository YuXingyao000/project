import os
import numpy as np
import argparse
from pathlib import Path

from tools.data.PartialPointCloudSampler import PartialPointCloudSampler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--radius", type=float, default=2)
    parser.add_argument("--elevations", type=list, default=[30, -30])
    parser.add_argument("--num_azimuths", type=int, default=8)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    
    viewpoints = PartialPointCloudSampler.generate_viewpoints(radius=args.radius, elevations=args.elevations, num_azimuths=args.num_azimuths)
    
    for data_id in os.listdir(data_root):
        data_path = data_root / data_id / f"{data_id}.ply"
        if not data_path.exists():
            continue
        if not (output_root / data_id).exists():
            os.makedirs(output_root / data_id)
            
        sampler = PartialPointCloudSampler(data_path)
        for i, viewpoint in enumerate(viewpoints):
            partial_point_clouds = []
            for n_points in [2048, 4096, 6144]:
                sampled_point_cloud = sampler.sample_partial_point_cloud(viewpoint, n_points=n_points)
                partial_point_clouds.append(sampled_point_cloud)

            np.savez(output_root / data_id / f"{data_id}_{i}.npz", 
                     simple=partial_point_clouds[0], 
                     moderate=partial_point_clouds[1], 
                     hard=partial_point_clouds[2])

                
    