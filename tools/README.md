# Data processing

> This data processing module is designed for the ABC dataset.

> [!NOTE] This is W.I.P.

## Get all the processed data
Following `ParseNet`, we just use the data file that contains at least 7 surfaces and at most 30. All data files that have more than 1 solid are removed.

To generate the dataset, run the following command:
```
python -m tools.process_abc \
    --data_root /path/to/abc_data \
    --output_root /path/to/processed_data \
    --batch_size 32 \
    --brep_sample_resolution 16 \
    --point_cloud_sample_num 8192 \
    --scanned_pc_n_points 2048 \
    --use_ray \
    --num_cpus 16
```

> [!NOTE]
> If you are using a headless server, you need to set the `DISPLAY` environment variable to `:99`.
> You can do this by running the following command:
> ```
> sudo apt-get update
> sudo apt-get install xvfb
> Xvbf :99 -screen 0 1920x1080x24 & export DISPLAY=:99
> ```