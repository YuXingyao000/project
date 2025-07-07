#! /bin/bash

python -m DataProcess.process_abc --data_root data --output_root processed_data --batch_size 32 --num_cpus 16 --brep_sample_resolution 32 --point_cloud_sample_num 8192 --use_ray