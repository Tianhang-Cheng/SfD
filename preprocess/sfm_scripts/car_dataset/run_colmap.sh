#!/bin/bash

# Run ORB-SLAM for a given snippet of a video
# Snippet folder structure:
# $DATASET_ROOT/
#   - images/
#       - car001/
#           - *.png
#   - run_data/
#       - car001/
#           - orb_slam.yaml
#           - image_0_reized (symlink to images/car001)
#           - image_names.txt
#           - timestamps.txt
#           - orb_output/

set -eu

if [ $# -ne 3 ]; then
    echo "Usage: ./run_colmap.sh /path/to/code/dir /path/to/dataset/dir log_id"
    exit 1
fi

function abs_path {
  (cd "$1" &>/dev/null && printf "%s" "$PWD")
}

# command line args
code_dir=$(abs_path "$1")
dataset_root=$(abs_path "$2")
echo "Code dir at $code_dir"
echo "Dataset dir at $dataset_root"
log_id="$3"

img_dir="${dataset_root}/images/${log_id}"
if [ ! -d "$img_dir" ]; then
    echo "Image directory ${img_dir} does not exist! Are you sure you entered the correct info? exiting..."
    exit 1
fi

data_dir="${dataset_root}/run_data/${log_id}"
out_dir="${data_dir}/colmap_direct_output"
if [ -d "$out_dir" ]; then
    rm -rf "$out_dir"
    # echo -n "${out_dir} is already a directory potentially containing ORB results. Do you want to remove it and continue? (y/n)? "
    # read answer
    # if [ "$answer" != "${answer#[Yy]}" ] ;then
    #     rm -rf "$out_dir"
    # else
    #     echo "You said no, so exiting..."
    #     exit 1
    # fi
fi
mkdir -p "$out_dir"

colmap automatic_reconstructor \
    --workspace_path "$out_dir" \
    --image_path "$img_dir" \
    --camera_model SIMPLE_PINHOLE \
    --single_camera 1

cd "$code_dir/scripts"
mkdir -p "${out_dir}/final"
python -m colmap_utils.read_write_model \
    --input_model "${out_dir}/sparse/0/" \
    --input_format .bin \
    --output_model "${out_dir}/final/" \
    --output_format .txt


python -m colmap_utils.read_write_model \
    --input_model "${colmap_out_dir}/dense/0/sparse/" \
    --input_format .bin \
    --output_model "${colmap_out_dir}/dense/0/sparse/" \
    --output_format .txt