#!/bin/bash

# Run ORB-SLAM for a given snippet of a video
# Snippet folder structure:
# $DATASET_ROOT/
#   - train/
#       - log_id1.txt
#       - log_id2.txt
#       - ...
#   - snippet/
#       - train/
#           - log_id1/
#               - snippet_id1/
#                   - times.txt
#                   - image_names.txt
#                   - orb_output/
#               - snippet_id2/
#               - ...
#           - ...

set -eu

if [ $# -ne 5 ]; then
    echo "Usage: ./run_orbslam.sh /path/to/code/dir /path/to/dataset/dir [train|validation|test] log_id snippet_id"
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
split_name="$3"
log_id="$4"
snippet_id="$5"

data_dir="${dataset_root}/data/${split_name}/${log_id}"
if [ ! -d "$data_dir" ]; then
    echo "Data directory ${data_dir} does not exist! Are you sure you entered the correct info? exiting..."
    exit 1
fi
snippet_dir="${dataset_root}/snippet/${split_name}/${log_id}/${snippet_id}"
if [ ! -d "$snippet_dir" ]; then
    echo "Snippet directory ${snippet_dir} does not exist! Are you sure you entered the correct info? exiting..."
    exit 1
fi

cd "$code_dir/scripts"
echo "Checking/Extracting the timings and image names of the provided snippet..."
python -m extract_snippet --dataset_root "$dataset_root" --split "$split_name" --log_id "$log_id" --snippet_id "$snippet_id"

cd "$snippet_dir"
out_dir="${snippet_dir}/orb_output"
if [ -d "$out_dir" ]; then
    echo -n "${out_dir} is already a directory potentially containing ORB results. Do you want to remove it and continue? (y/n)? "
    read answer
    if [ "$answer" != "${answer#[Yy]}" ] ;then
        rm -rf "$out_dir"
    else
        echo "You said no, so exiting..."
        exit 1
    fi
fi
mkdir -p "$out_dir"

cd "$code_dir/ORB_SLAM2"
echo "Running ORB-SLAM, with output written to ${out_dir}"
./Examples/Monocular/mono_kitti \
    Vocabulary/ORBvoc.txt \
    "${data_dir}/orb_slam.yaml" \
    "${data_dir}" \
    "${snippet_dir}" \
    "${out_dir}" 2>&1 | tee "${out_dir}/log.txt"
echo "ORB-SLAM finished successfully"