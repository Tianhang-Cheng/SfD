
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
os.makedirs(colmap_dir, exist_ok=True)


################################# Load estimated ORB trajectory ###################################
traj_proj, intrinsics_proj, _, proj_mats = read_reality_capture_trajectory_from_proj_mat(
    os.path.join(snippet_dir, "reality_capture_output", "P_matrix"),
    os.path.join(snippet_dir, "colmap_output", "times.txt"),
    os.path.join(snippet_dir, "colmap_output", "image_names.txt"), transpose=True, convert_to_colmap=False)

colmap_imgs = sorted([img_name for img_name in os.listdir(os.path.join(snippet_dir, "reality_capture_output", "P_matrix")) if img_name.endswith(".png")])

img_dest_dir = os.path.join(colmap_dir, "images")
os.makedirs(img_dest_dir, exist_ok=True)
for img_name in colmap_imgs:
    if not os.path.exists(os.path.join(img_dest_dir, img_name)):
        os.symlink(os.path.join(snippet_dir, "reality_capture_output", "P_matrix", img_name), os.path.join(img_dest_dir, img_name))

# find the poses corresponding to the colmap times
frame_ids = list(range(len(colmap_imgs)))
assert len(traj_proj.timestamps) == len(frame_ids)


######################## Generate cameras.txt, images.txt, points_3d.txt ##########################
colmap_input_dir = os.path.join(colmap_dir, "input")
os.makedirs(colmap_input_dir, exist_ok=True)

# cameras.txt
img_sample = cv2.imread(os.path.join(snippet_dir, "reality_capture_output", "P_matrix", "frame000001.png"))
width = img_sample.shape[1]
height = img_sample.shape[0]
with open(os.path.join(colmap_input_dir, "cameras.txt"), "w") as f:
    for idx, K in enumerate(intrinsics_proj):
        f.write("{} PINHOLE {} {} {} {} {} {}\n".format(idx + 1, width, height, K[0,0], K[1,1], K[0,2], K[1,2]))

# images.txt
# load the database and find the image id to image name mapping
# image_name_to_id = get_colmap_image_name_to_id_mapping(database_fpath)

# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 2, mean observations per image: 2
# see https://colmap.github.io/format.html and https://colmap.github.io/faq.html
with open(os.path.join(colmap_input_dir, "images.txt"), "w") as f:
    for i, (pose, img_name) in enumerate(zip(traj_proj.poses_se3, colmap_imgs)):
        # convert the orb-slam pose (camera to world frame) to COLMAP format
        ext = np.linalg.inv(pose)
        qvec = rotmat2qvec(ext[:3, :3])
        f.write("{} {} {} {} {} {} {} {} {} {}\n".format(
            i+1, qvec[0], qvec[1], qvec[2], qvec[3],
            ext[0,3], ext[1,3], ext[2,3], i+1, img_name))
        f.write("\n")  # empty line for keypoints

# points_3d.txt
# should be an empty file
open(os.path.join(colmap_input_dir, "points3D.txt"), "a").close()
