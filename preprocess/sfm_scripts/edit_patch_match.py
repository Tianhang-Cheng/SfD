
"""Convert ORB-SLAM output into COLMAP formats. Prepare images.txt, camera.txt and points3D.txt"""
import cv2
import os
import numpy as np
import argparse

from evo.core.trajectory import PoseTrajectory3D

from file_utils import get_intrinsics, read_timestamps, read_str_file, read_tum_trajectory, \
    align_timestamps, get_colmap_image_name_to_id_mapping, read_reality_capture_trajectory_from_proj_mat
from geo_utils import rotmat2qvec


####################################### Parse Args ################################################
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root directory")
parser.add_argument("--split", type=str, required=True, choices=["train", "validation", "test"],
    help="Dataset split, one of [train|validation|test]")
parser.add_argument("--log_id", type=str, required=True, help="Log id")
parser.add_argument("--snippet_id", type=str, required=True, help="Snippet id for the video snippet")
args = parser.parse_args()

snippet_dir = os.path.join(args.dataset_root, "snippet", args.split, args.log_id, args.snippet_id)
assert(os.path.isdir(snippet_dir))
colmap_dir = os.path.join(snippet_dir, "reality_poses_colmap_output")
patch_match_filename = os.path.join(colmap_dir, "dense", "stereo", "patch-match.cfg")

################################# Load estimated ORB trajectory ###################################
colmap_imgs = sorted([img_name for img_name in os.listdir(os.path.join(snippet_dir, "reality_capture_output", "P_matrix")) if img_name.endswith(".png")])
traj_proj, intrinsics_proj, _, proj_mats = read_reality_capture_trajectory_from_proj_mat(
    os.path.join(snippet_dir, "reality_capture_output", "P_matrix"),
    os.path.join(snippet_dir, "colmap_output", "times.txt"),
    os.path.join(snippet_dir, "colmap_output", "image_names.txt"), transpose=True, convert_to_colmap=False)

with open(patch_match_filename, "w") as f:
    for idx, img_name in enumerate(colmap_imgs):
        f.write(f"{img_name}\n")
        img_names = []
        for neighbor_idx in range(max(0, idx - 10), min(idx + 11, len(colmap_imgs))):
            if idx == neighbor_idx:
                continue
            if np.linalg.norm(traj_proj.poses_se3[idx][:3, 3] - traj_proj.poses_se3[neighbor_idx][:3, 3]) <= 2.0:
                img_names.append(colmap_imgs[neighbor_idx])
        f.write("{}\n".format(",".join(img_names)))
        
