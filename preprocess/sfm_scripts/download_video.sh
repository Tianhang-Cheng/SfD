#!/bin/bash

# Script to download data (video, images, times) with metadata from the MannequinChallenge Dataset
# Dataset metadata can be downloaded at https://google.github.io/mannequinchallenge/www/download.html
# The dataset only contains the URLs and GT intrinsics, times and poses for a short 1-2s sequence with the following structure:
# $DATASET_ROOT
#   - train
#       - log_id1.txt
#       - log_id2.txt
#       - ...
#   - test
#       - ...
#   - validation
#       - ...
#   - data
#       - train
#           - log_id1
#           - log_id2
#           - ...
#       - test
#       - validation
# This script downloads the video and extracts image frames and capture times from a given txt file.
# The data will be saved at ${dataset_root}/data/${split_name}/${log_id}

set -eu
# set -o pipefail # comment out since we would like to ignore SIGPIPE errors (error code 141)

if [ $# -ne 4 ]; then
    echo "Usage: ./download_video.sh /path/to/code/dir /path/to/dataset/dir [train|validation|test] log_id"
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

out_dir="${dataset_root}/data/${split_name}/${log_id}"
if [ -d "$out_dir" ]; then
    echo -n "${out_dir} is already a directory. Do you want to remove it and continue? (y/n)? "
    read answer
    if [ "$answer" != "${answer#[Yy]}" ] ;then
        rm -rf "$out_dir"
    else
        echo "You said no, so exiting..."
        exit 1
    fi
fi
mkdir -p "$out_dir"
echo "Saving video and images data to ${out_dir}"
cd "$out_dir"

# download the youtube video
meta_data="${dataset_root}/${split_name}/${log_id}.txt"
echo "Reading URL from ${meta_data}"
cat "${meta_data}" | head -1
url=$(cat "${meta_data}" | head -1)
echo "Downloading youtube video at url ${url}"
youtube-dl -o "%(title)s.%(ext)s" "$url"
# vid_fname=$(youtube-dl --get-filename -o "%(title)s.%(ext)s" "$url")
vid_fname=$(ls | head -1)  # hack, since sometimes youtube-dl --get-filename gets .mp4 but video gets downloaded as mkv
echo "Video downloaded as ${out_dir}/${vid_fname}"

# extract the images, times, and downsample them
mkdir image_0
echo "Extracting image frames into ${out_dir}/image_0, and writing times to ${out_dir}/times_all.txt"
# we use opencv to extract frame times, so for consistency we also use it for image extraction.
# alternatively, we can run ffmpeg -i "$vid_fname" -start_number 0 image_0/%06d.png to extract only the image frames
cd "${code_dir}/scripts"
python -m extract_image_frames --data_dir "$out_dir" --video_name "$vid_fname"
echo "Image frame extraction completed."
cd "$out_dir"
convert image_0/000000.png -print "Image size: %wx%h\n" /dev/null
mkdir image_0_resized

echo "Downsample images by 2x to ${out_dir}/image_0_resized"
cd image_0
find . -iname "*.png" | xargs -L1 -I{} convert -resize 50%  "{}" ../image_0_resized/"{}"
cd ..
convert image_0_resized/000000.png -print "Downsampled image size: %wx%h\n" /dev/null

width=$(identify -format '%w' image_0_resized/000000.png)
height=$(identify -format '%h' image_0_resized/000000.png)
cd "${code_dir}/scripts"
echo "Dumping ORB-SLAM config yaml files"
python -m dump_orb_slam_config --img_width $width --img_height $height --dataset_root "$dataset_root" --split "$split_name" --log_id "$log_id"
python -m dump_orb_slam_config --img_width $width --img_height $height --dataset_root "$dataset_root" --split "$split_name" --log_id "$log_id" --use_original_setting

echo "Succecss!"