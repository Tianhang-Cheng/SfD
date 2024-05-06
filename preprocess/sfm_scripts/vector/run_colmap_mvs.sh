#!/bin/bash

# Run ORB-SLAM for a given snippet of a video
# Snippet folder structure:
# $DATASET_ROOT/
#   - snippet/
#       - train/
#           - log_id1/
#               - snippet_id1/
#                   - times.txt
#                   - image_names.txt
#                   - orb_output/
#                   - colmap_output/
#               - snippet_id2/
#               - ...
#           - ...
#       - ...

set -eu

if [ $# -ne 5 ]; then
    echo "Usage: ./run_colmap.sh /path/to/code/dir /path/to/dataset/dir [train|validation|test] log_id snippet_id"
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

# Check if all required directories exist
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

colmap_out_dir="${snippet_dir}/reality_poses_colmap_output"

# Prepare COLMAP image files from the ORB-SLAM output
cd "$code_dir/scripts"

# colmap image_undistorter \
#     --image_path "${colmap_out_dir}/images/" \
#     --input_path "${colmap_out_dir}/input/" \
#     --output_path "${colmap_out_dir}/dense/" \
#     --output_type COLMAP \
#     --max_image_size 1600

python -m edit_patch_match --dataset_root "$data_dir" --split "$split_name" --log_id "$log_id" --snippet_id "$snippet_id"

colmap patch_match_stereo \
    --workspace_path "${colmap_out_dir}/dense/" \
    --workspace_format COLMAP \
    --PatchMatchStereo.window_radius 4 \
    --PatchMatchStereo.num_samples 15 \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.geom_consistency_regularizer 1.0 \
    --PatchMatchStereo.geom_consistency_max_cost 1.5 \
    --PatchMatchStereo.filter_min_ncc 0.2 \
    --PatchMatchStereo.filter_min_num_consistent 3 \
    --PatchMatchStereo.depth_min 0.01 \
    --PatchMatchStereo.depth_max 100
echo "Finished MVS"

python -m colmap_utils.read_write_model \
    --input_model "${colmap_out_dir}/dense/sparse/" \
    --input_format .bin \
    --output_model "${colmap_out_dir}/dense/sparse/" \
    --output_format .txt

colmap stereo_fusion \
   --workspace_path "${colmap_out_dir}/dense/" \
   --workspace_format COLMAP \
   --input_type geometric \
   --output_path "${colmap_out_dir}/dense/fused.ply"
colmap poisson_mesher \
   --input_path "${colmap_out_dir}/dense/fused.ply" \
   --output_path "${colmap_out_dir}/dense/meshed-poisson.ply"
# colmap delaunay_mesher \
#    --input_path "${colmap_out_dir}/dense/" \
#    --input_type dense
#    --output_path "${colmap_out_dir}/dense/meshed-delaunay.ply"

# convert .bin output to .txt for readability

echo "COLMAP finished successfully"
