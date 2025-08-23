#!/bin/bash

python -m tools.process_abc \
    --data_root /mnt/d/data/DeepCAD/organized_data/ \
    --output_root /mnt/d/data/processed_deepcad \
    --brep_sample_resolution 16 \
    --point_cloud_sample_num 8192 \
    --scanned_pc_n_points 8196 \
    --batch_size 32 \
    --num_cpus 96 \
    --use_ray