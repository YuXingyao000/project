#! /bin/bash

python -m tools.create_partial_point_cloud --data_root processed_data --output_root processed_data_partial --radius 2 --elevation 45 --num_azimuths 4 --batch_size 32 --use_ray --is_evaluation