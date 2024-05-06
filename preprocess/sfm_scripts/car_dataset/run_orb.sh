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
    echo "Usage: ./run_orbslam.sh /path/to/code/dir /path/to/dataset/dir log_id"
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
echo "Preparing ORB input..."
if [ ! -e "${data_dir}/orb_slam_60fov.yaml" ]; then
    mkdir -p "$data_dir"
    cd "$code_dir/scripts"
    python -m car_dataset.prepare_orb_input --dataset_root "$dataset_root" --log_id "$log_id"
fi

out_dir="${data_dir}/orb_output"
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

cd "$code_dir/ORB_SLAM2"
echo "Running ORB-SLAM, with output written to ${out_dir}"
./Examples/Monocular/mono_kitti \
    Vocabulary/ORBvoc.txt \
    "${data_dir}/orb_slam_60fov.yaml" \
    "${data_dir}" \
    "${data_dir}" \
    "${out_dir}" 2>&1 | tee "${out_dir}/log.txt"
echo "ORB-SLAM finished successfully"