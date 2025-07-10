#! /bin/bash

python -m tools.process_abc --data_root /mnt/d/data/test_data --output_root /mnt/d/data/processed_test_data --batch_size 32 --brep_sample_resolution 32 --point_cloud_sample_num 8192 --use_ray