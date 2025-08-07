import os
import ray
from pathlib import Path
import argparse
import traceback
import shutil
import sys

from tools.data import SolidProcessor, ABCReader

MAX_SURFACE_NUM = 30
MIN_SURFACE_NUM = 7

def process_step_folder(data_root, output_root, step_ids, brep_sample_resolution, point_cloud_sample_num):
    count = 0
    for step_id in step_ids:
        try:
            if not os.path.exists(output_root / step_id):
                os.makedirs(output_root / step_id)
            reader = ABCReader()
            reader.read_step_file(data_root / step_id / os.listdir(data_root / step_id)[0]) # Only one step file in each ABC folder
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
            count += 1
            print(f"Processed {count} steps")
            if count > 1000:
                break
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
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--use_ray", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_cpus", type=int, default=16)
    parser.add_argument("--brep_sample_resolution", type=int, default=32)
    parser.add_argument("--point_cloud_sample_num", type=int, default=8192)
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    batch_size = args.batch_size
    
    step_ids = os.listdir(data_root)
    step_ids.sort()
    
    if args.use_ray:
        # Start Ray without dashboard to avoid Windows issues
        context = ray.init()
        print(f"Ray initialized with {context.dashboard_url} (dashboard disabled)")
        
        process_step_folder_remote = ray.remote(process_step_folder)
        tasks = []
        
        for i in range(0, len(step_ids), batch_size):
            batch_ids = step_ids[i:min(len(step_ids), i + batch_size)]
            tasks.append(process_step_folder_remote.remote(data_root, output_root, batch_ids, args.brep_sample_resolution, args.point_cloud_sample_num))
        ray.get(tasks)
        ray.shutdown()
    else:
        process_step_folder(data_root, output_root, step_ids, args.brep_sample_resolution, args.point_cloud_sample_num)
