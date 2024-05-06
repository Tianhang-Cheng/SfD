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

orb_out_dir="${data_dir}/orb_output"
if [ ! -d "$orb_out_dir" ]; then
    echo "Data directory ${orb_out_dir} does not exist! COLMAP can only be run with ORB-SLAM output. \
        Are you sure you entered the correct info? exiting..."
    exit 1
fi
colmap_out_dir="${data_dir}/colmap_orb_output"
if [ -d "$colmap_out_dir" ]; then
    rm -rf "$colmap_out_dir"
    # echo -n "${colmap_out_dir} is already a directory potentially containing COLMAP results. Do you want to remove it and continue? (y/n)? "
    # read answer
    # if [ "$answer" != "${answer#[Yy]}" ] ;then
    #     rm -rf "$colmap_out_dir"
    # else
    #     echo "You said no, so exiting..."
    #     exit 1
    # fi
fi

# Prepare COLMAP image files from the ORB-SLAM output
cd "$code_dir/scripts"
echo "Preparing COLMAP image files"
python -m car_dataset.gen_colmap_images \
    --dataset_root "$dataset_root" \
    --log_id "$log_id"

# Run COLMAP
mkdir "${colmap_out_dir}/triangulated"
mkdir "${colmap_out_dir}/final"
mkdir "${colmap_out_dir}/dense"
echo "Running COLMAP, with output written to ${colmap_out_dir}"
colmap feature_extractor \
    --ImageReader.single_camera 1 \
    --ImageReader.default_focal_length_factor 0.85 \
    --SiftExtraction.peak_threshold 0.02 \
    --database_path "${colmap_out_dir}/database.db" \
    --image_path "${colmap_out_dir}/images/"
echo "Finished feature extraction"
colmap exhaustive_matcher \
    --database_path "${colmap_out_dir}/database.db" \
    --SiftMatching.max_error 3 \
    --SiftMatching.min_inlier_ratio 0.3 \
    --SiftMatching.min_num_inliers 30 \
    --SiftMatching.guided_matching 1

echo "Generating COLMAP sparse input files"
python -m car_dataset.gen_colmap_input \
    --dataset_root "$dataset_root" \
    --log_id "$log_id"

colmap point_triangulator \
    --Mapper.tri_merge_max_reproj_error 3 \
    --Mapper.ignore_watermarks 1 \
    --Mapper.filter_max_reproj_error 2 \
    --database_path "${colmap_out_dir}/database.db" \
    --image_path "${colmap_out_dir}/images/" \
    --input_path "${colmap_out_dir}/input/" \
    --output_path "${colmap_out_dir}/triangulated/"
colmap bundle_adjuster \
    --input_path "${colmap_out_dir}/triangulated/" \
    --output_path "${colmap_out_dir}/final/"
colmap image_undistorter \
    --image_path "${colmap_out_dir}/images/" \
    --input_path "${colmap_out_dir}/final/" \
    --output_path "${colmap_out_dir}/dense/" \
    --output_type COLMAP \
    --max_image_size 960

# convert .bin output to .txt for readability
python -m colmap_utils.read_write_model \
    --input_model "${colmap_out_dir}/final/" \
    --input_format .bin \
    --output_model "${colmap_out_dir}/final/" \
    --output_format .txt

python -m colmap_utils.read_write_model \
    --input_model "${colmap_out_dir}/dense/sparse/" \
    --input_format .bin \
    --output_model "${colmap_out_dir}/dense/sparse/" \
    --output_format .txt

colmap patch_match_stereo \
    --workspace_path "${colmap_out_dir}/dense/" \
    --workspace_format COLMAP \
    --PatchMatchStereo.window_radius 4 \
    --PatchMatchStereo.num_samples 15 \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.geom_consistency_regularizer 1.0 \
    --PatchMatchStereo.geom_consistency_max_cost 1.5 \
    --PatchMatchStereo.filter_min_ncc 0.2 \
    --PatchMatchStereo.filter_min_num_consistent 3
echo "Finished MVS"
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

echo "COLMAP finished successfully"
