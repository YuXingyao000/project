import os
import ray
from pathlib import Path
import argparse
import traceback
import shutil
import sys
import numpy as np
import random

# Set DISPLAY environment variable for headless servers
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':99'

from tools.data import SolidProcessor, ABCReader

MAX_SURFACE_NUM = 30
MIN_SURFACE_NUM = 7

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")

def process_step_folder(data_root, output_root, step_ids, brep_sample_resolution, point_cloud_sample_num, scanned_pc_n_points, base_seed=None):
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':99'
    
    for i, step_id in enumerate(step_ids):
        if base_seed is not None:
            file_seed = base_seed + i
            set_random_seed(file_seed)
        try:
            if not os.path.exists(output_root / step_id):
                os.makedirs(output_root / step_id)
            reader = ABCReader()
            reader.read_step_file(data_root / step_id / "normalized_shape.step") # Only one step file in each ABC folder
            solids = reader.solids
            if len(solids) > 1:
                raise ValueError(f"Step {step_id} has {len(solids)} solids, skipping")
            if len(solids) == 0:
                raise ValueError(f"Step {step_id} has no solids, skipping")
            solid_processor = SolidProcessor(solids[0])
            if len(solid_processor.get_surfaces()) > MAX_SURFACE_NUM or len(solid_processor.get_surfaces()) < MIN_SURFACE_NUM:
                raise ValueError(f"Step {step_id} has {len(solid_processor.get_surfaces())} surfaces, skipping")
            solid_processor.normalize_shape()
            solid_processor.export_solid(output_root / step_id / f"{step_id}.step")
            solid_processor.export_mesh(output_root / step_id / f"{step_id}.stl")
            solid_processor.export_point_cloud(output_root / step_id / f"{step_id}.ply", sample_num=point_cloud_sample_num)
            solid_processor.export_uv_grids(output_root / step_id / f"{step_id}.npz", sample_resolution=brep_sample_resolution)
            solid_processor.export_point_cloud_numpy(output_root / step_id / f"{step_id}_pc.npz")
            solid_processor.export_random_cropped_pc(output_root / step_id / f"{step_id}_cropped_pc.h5")
            solid_processor.export_scanned_point_cloud(output_root / step_id / f"{step_id}_scanned_pc_cube.h5", strategy='cube', n_points=scanned_pc_n_points)
            # solid_processor.export_scanned_point_cloud(output_root / step_id / f"{step_id}_scanned_pc_sphere.h5", strategy='sphere', n_points=scanned_pc_n_points)
            # solid_processor.export_photos(output_root / step_id / f"{step_id}_photos.npz")
        except Exception as e:
            with open(output_root / "error.txt", "a") as f:
                tb_list = traceback.extract_tb(sys.exc_info()[2])
                last_traceback = tb_list[-1]
                f.write(step_id + ": " + str(e) + "\n")
                f.write(f"An error occurred on line {last_traceback.lineno} in {last_traceback.name}\n\n")
                print(e)
                shutil.rmtree(output_root / step_id)
            continue



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to the data root")
    parser.add_argument("--output_root", type=str, required=True, help="Path to the output root")
    parser.add_argument("--use_ray", action="store_true", help="Use Ray for parallel processing")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for parallel processing")
    parser.add_argument("--num_cpus", type=int, default=16, help="Number of CPUs for parallel processing")
    parser.add_argument("--brep_sample_resolution", type=int, default=32, help="Brep sample resolution")
    parser.add_argument("--point_cloud_sample_num", type=int, default=8192, help="Number of points to sample a whole point cloud")
    parser.add_argument("--scanned_pc_n_points", type=int, default=2048, help="Number of points to sample a point cloud with a virtual scanner")
    parser.add_argument("--random_seed", type=int, default=114514, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    batch_size = args.batch_size
    set_random_seed(args.random_seed)
    
    step_ids = os.listdir(data_root)
    step_ids.sort()
    
    if args.use_ray:
        # Start Ray without dashboard to avoid Windows issues
        context = ray.init()
        print(f"Ray initialized with {context.dashboard_url} (dashboard disabled)")
        
        process_step_folder_remote = ray.remote(process_step_folder)
        tasks = []
        
        for i in range(0, len(step_ids), batch_size):
            base_seed = args.random_seed + i
            batch_ids = step_ids[i:min(len(step_ids), i + batch_size)]
            tasks.append(process_step_folder_remote.remote(data_root, output_root, batch_ids, args.brep_sample_resolution, args.point_cloud_sample_num, args.scanned_pc_n_points, base_seed))
        ray.get(tasks)
        ray.shutdown()
    else:
        process_step_folder(data_root, output_root, step_ids, args.brep_sample_resolution, args.point_cloud_sample_num, args.scanned_pc_n_points, args.random_seed)
