import argparse
import os
import numpy as np

from file_utils import read_raw_colmap_camera, read_raw_colmap_trajectory, read_reality_capture_trajectory, read_reality_capture_trajectory_from_proj_mat


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root directory")
parser.add_argument("--split", type=str, required=True, choices=["train", "validation", "test"],
    help="Dataset split, one of [train|validation|test]")
parser.add_argument("--log_id", type=str, default=None, help="Log id")
parser.add_argument("--snippet_id", type=str, default=None, help="Snippet id for the video snippet")
parser.add_argument("--pose_type", type=str, choices=["colmap", "reality", "both"], default="both", help="Type of traj to dump")
args = parser.parse_args()

if args.log_id is None:
    log_ids = os.listdir(os.path.join(args.dataset_root, "snippet", args.split))
else:
    log_ids = [args.log_id]

for log_id in log_ids:
    log_dir = os.path.join(args.dataset_root, "snippet", args.split, log_id)
    if not os.path.isdir(log_dir):
        print(f"log dir {log_dir} is not a directory, skipping...")
        continue

    if args.snippet_id is None:
        snippet_ids = os.listdir(log_dir)
    else:
        snippet_ids = [args.snippet_id]
    
    for snippet_id in snippet_ids:
        snippet_dir = os.path.join(log_dir, snippet_id)
        print(f"Processing log_id {log_id}, snippet_id {snippet_id}")
        colmap_dir = os.path.join(snippet_dir, "colmap_output")
        reality_dir = os.path.join(snippet_dir, "reality_capture_output")
        if (not os.path.isdir(colmap_dir)) and (not os.path.isdir(reality_dir)):
            print(f"{colmap_dir} is not a directory, skipping...")
            continue

        if args.pose_type in ["colmap", "both"]:
            traj, image_names = read_raw_colmap_trajectory(
                os.path.join(snippet_dir, "colmap_output", "dense", "sparse", "images.txt"),
                os.path.join(snippet_dir, "colmap_output", "times.txt"),
                os.path.join(snippet_dir, "colmap_output", "image_names.txt"), return_images=True, return_extrinsics=True)
            traj_sanity_check = read_raw_colmap_trajectory(
                os.path.join(snippet_dir, "colmap_output", "refined", "images.txt"),
                os.path.join(snippet_dir, "colmap_output", "times.txt"),
                os.path.join(snippet_dir, "colmap_output", "image_names.txt"), return_extrinsics=True)
            for pose1, pose2 in zip(traj.poses_se3, traj_sanity_check.poses_se3):
                assert np.allclose(pose1, pose2)

            # colamp trajectory shares the same camera
            K = read_raw_colmap_camera(os.path.join(snippet_dir, "colmap_output", "dense", "sparse", "cameras.txt"))

            with open(os.path.join(snippet_dir, "colmap_GT_poses.txt"), "w") as f:
                for image_name, time, pose in zip(image_names, traj.timestamps, traj.poses_se3):
                    content = list(map(str, [image_name, float(time)] + pose.ravel().tolist() + K.ravel().tolist()))
                    f.write("{}\n".format(" ".join(content)))
        
        if args.pose_type in ["reality", "both"]:
            traj_euler, intrinsics_euler, traj_img_names_euler = read_reality_capture_trajectory(
                os.path.join(snippet_dir, "reality_capture_output", "poses.csv"),
                os.path.join(snippet_dir, "colmap_output", "times.txt"),
                os.path.join(snippet_dir, "colmap_output", "image_names.txt"),
                os.path.join(snippet_dir, "reality_capture_output", "P_matrix"),
                # os.path.join(snippet_dir, "colmap_output", "dense", "images"),
                convert_to_colmap=True,
                transpose=True,
                return_extrinsics=True)

            traj_proj, intrinsics_proj, traj_img_names_proj, _ = read_reality_capture_trajectory_from_proj_mat(
                os.path.join(snippet_dir, "reality_capture_output", "P_matrix"),
                os.path.join(snippet_dir, "colmap_output", "times.txt"),
                os.path.join(snippet_dir, "colmap_output", "image_names.txt"),
                convert_to_colmap=True,
                transpose=True,
                return_extrinsics=True)
            
            with open(os.path.join(snippet_dir, "reality_GT_poses.txt"), "w") as f:
                for image_name, time, pose, K in zip(traj_img_names_proj, traj_proj.timestamps, traj_proj.poses_se3, intrinsics_euler):
                    content = list(map(str, [image_name, float(time)] + pose.ravel().tolist() + K.ravel().tolist()))
                    f.write("{}\n".format(" ".join(content)))
