
"""Prepare the images input to COLMAP based on successful ORB-SLAM frames."""
import cv2
import os
import numpy as np
import argparse

from shutil import copyfile
from evo.core.sync import associate_trajectories

from file_utils import read_gt_trajectory, read_tum_trajectory, read_timestamps, \
    read_str_file, align_timestamps


####################################### Parse Args ################################################
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root directory")
parser.add_argument("--log_id", type=str, required=True, help="Log id")
args = parser.parse_args()

data_dir = os.path.join(args.dataset_root, "images", args.log_id)
assert(os.path.isdir(data_dir))
log_dir = os.path.join(args.dataset_root, "run_data", args.log_id)
assert(os.path.isdir(log_dir))
orb_result_dir = os.path.join(log_dir, "orb_output")
assert(os.path.isdir(orb_result_dir))
output_dir = os.path.join(log_dir, "colmap_orb_output")
os.makedirs(output_dir)  # we assume the output_dir doesn't exist


######################################### Prepare Images ##########################################
# load the ORB-SLAM trajectory
orb_traj_fpath = os.path.join(orb_result_dir, "KeyFrameTrajectory.txt")
assert(os.path.isfile(orb_traj_fpath))
traj_orb = read_tum_trajectory(orb_traj_fpath)

# find the image filenames corresponding to the timestamps
times_all = read_timestamps(os.path.join(log_dir, "times.txt"))
images_all = read_str_file(os.path.join(log_dir, "image_names.txt"))
i = 0
j = 0
frame_ids = []
etol = 1  # microseconds
frame_ids = align_timestamps(times_all, traj_orb.timestamps, etol)
assert(len(frame_ids) == len(traj_orb.timestamps))
# write the times and image names
with open(os.path.join(output_dir, "times.txt"), "w") as f:
    for idx in frame_ids:
        f.write(f"{times_all[idx]}\n")
with open(os.path.join(output_dir, "image_names.txt"), "w") as f:
    for idx in frame_ids:
        f.write(f"{images_all[idx]}\n")

# symlink the corresponding images into a new folder
img_src_dir = os.path.join(data_dir)
img_dest_dir = os.path.join(output_dir, "images")
os.makedirs(img_dest_dir)
for idx in frame_ids:
    os.symlink(os.path.join(img_src_dir, images_all[idx]), os.path.join(img_dest_dir, images_all[idx]))
