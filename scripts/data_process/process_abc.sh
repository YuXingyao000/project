#!/bin/bash

python -m tools.process_abc \
    --data_root /mnt/d/data/data_step \
    --output_root /mnt/d/data/data_step_processed \
    --brep_sample_resolution 16 \
    --point_cloud_sample_num 8192 \
    --scanned_pc_n_points 2048 \
    --batch_size 32 \
    --num_cpus 96 \
    --use_ray