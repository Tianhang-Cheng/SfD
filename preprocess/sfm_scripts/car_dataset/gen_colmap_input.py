
"""Convert ORB-SLAM output into COLMAP formats. Prepare images.txt, camera.txt and points3D.txt"""
import cv2
import os
import numpy as np
import argparse

from evo.core.trajectory import PoseTrajectory3D

from file_utils import get_intrinsics, read_timestamps, read_str_file, read_tum_trajectory, \
    align_timestamps, get_colmap_image_name_to_id_mapping
from geo_utils import rotmat2qvec


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
colmap_dir = os.path.join(log_dir, "colmap_orb_output")
assert(os.path.isdir(colmap_dir))
database_fpath = os.path.join(colmap_dir, "database.db")
assert(os.path.isfile(database_fpath))


################################# Load estimated ORB trajectory ###################################
orb_traj_fpath = os.path.join(orb_result_dir, "KeyFrameTrajectory.txt")
assert(os.path.isfile(orb_traj_fpath))
traj_orb = read_tum_trajectory(orb_traj_fpath)

colmap_times = read_timestamps(os.path.join(colmap_dir, "times.txt"))
colmap_imgs = read_str_file(os.path.join(colmap_dir, "image_names.txt"))

# find the poses corresponding to the colmap times
frame_ids = align_timestamps(traj_orb.timestamps, colmap_times, 1)
traj_orb = PoseTrajectory3D(
    timestamps=colmap_times,
    poses_se3=[np.copy(traj_orb.poses_se3[idx]) for idx in frame_ids])


######################## Generate cameras.txt, images.txt, points_3d.txt ##########################
colmap_input_dir = os.path.join(colmap_dir, "input")
os.makedirs(colmap_input_dir)

# cameras.txt
img_sample = cv2.imread(os.path.join(data_dir, list(os.listdir(data_dir))[0]))
width = float(img_sample.shape[1])
height = float(img_sample.shape[0])
fx, fy, cx, cy = get_intrinsics("", width, height, True)
with open(os.path.join(colmap_input_dir, "cameras.txt"), "w") as f:
    f.write("1 PINHOLE {} {} {} {} {} {}\n".format(width, height, fx, fy, cx, cy))

# images.txt
# load the database and find the image id to image name mapping
image_name_to_id = get_colmap_image_name_to_id_mapping(database_fpath)

# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 2, mean observations per image: 2
# see https://colmap.github.io/format.html and https://colmap.github.io/faq.html
with open(os.path.join(colmap_input_dir, "images.txt"), "w") as f:
    for i, (pose, img_name) in enumerate(zip(traj_orb.poses_se3, colmap_imgs)):
        # convert the orb-slam pose (camera to world frame) to COLMAP format
        ext = np.linalg.inv(pose)
        qvec = rotmat2qvec(ext[:3, :3])
        f.write("{} {} {} {} {} {} {} {} {} {}\n".format(
            image_name_to_id[img_name], qvec[0], qvec[1], qvec[2], qvec[3],
            ext[0,3], ext[1,3], ext[2,3], 1, img_name))
        f.write("\n")  # empty line for keypoints

# points_3d.txt
# should be an empty file
open(os.path.join(colmap_input_dir, "points3D.txt"), "a").close()
