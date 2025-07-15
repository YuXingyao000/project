#!/bin/bash
SOURCE_DIR="/mnt/d/data/processed_test_data_partial/simple"
DEST_DIR="/mnt/d/data/processed_test_data_simple_xyz"

mkdir -p $DEST_DIR

for folder in $SOURCE_DIR/*/; do
    echo $folder
    file_number=$(basename $folder)
    if [ -d $folder ]; then
        target_file="$DEST_DIR/$file_number.xyz"
        ply_file="$folder${file_number}_0.ply"
        python tools/ply2xyz.py --ply_file "$ply_file" --xyz_file "$target_file"
    fi
done