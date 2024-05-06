
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
parser.add_argument("--split", type=str, required=True, choices=["train", "validation", "test"],
    help="Dataset split, one of [train|validation|test]")
parser.add_argument("--log_id", type=str, required=True, help="Log id")
parser.add_argument("--snippet_id", type=str, required=True, help="Snippet id for the video snippet")
parser.add_argument("--align_with_gt", action="store_true", default=False,
    help="Only use the frames aligned with GT. This is helpful when comparing with the original setting")
args = parser.parse_args()

data_dir = os.path.join(args.dataset_root, "data", args.split, args.log_id)
assert(os.path.isdir(data_dir))
snippet_dir = os.path.join(args.dataset_root, "snippet", args.split, args.log_id, args.snippet_id)
assert(os.path.isdir(snippet_dir))
orb_result_dir = os.path.join(snippet_dir, "orb_output")
assert(os.path.isdir(orb_result_dir))
output_dir = os.path.join(snippet_dir, "colmap_output")
os.makedirs(output_dir)  # we assume the output_dir doesn't exist


######################################### Prepare Images ##########################################
# load the ORB-SLAM trajectory
orb_traj_fpath = os.path.join(orb_result_dir, "KeyFrameTrajectory.txt")
assert(os.path.isfile(orb_traj_fpath))
traj_orb = read_tum_trajectory(orb_traj_fpath)
if args.align_with_gt:
    traj_gt = read_gt_trajectory(os.path.join(args.dataset_root, args.split, f"{args.log_id}.txt"))
    etol = min(1000, max(0, (traj_orb.timestamps[1] - traj_orb.timestamps[0]) / 2))  # microseconds
    _, traj_orb = associate_trajectories(traj_gt, traj_orb, max_diff=etol)

# find the image filenames corresponding to the timestamps
times_all = read_timestamps(os.path.join(snippet_dir, "times.txt"))
images_all = read_str_file(os.path.join(snippet_dir, "image_names.txt"))
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
img_src_dir = os.path.join(data_dir, "image_0")
img_dest_dir = os.path.join(output_dir, "images")
os.makedirs(img_dest_dir)
for idx in frame_ids:
    os.symlink(os.path.join(img_src_dir, images_all[idx]), os.path.join(img_dest_dir, images_all[idx]))
    # copyfile(os.path.join(img_src_dir, images_all[idx]), os.path.join(img_dest_dir, images_all[idx]))
