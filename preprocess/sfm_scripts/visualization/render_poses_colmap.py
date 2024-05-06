import trimesh
import pyrender
import numpy as np
import os
import argparse
import cv2
import matplotlib.pyplot as plt
from evo.core.sync import associate_trajectories

from colmap_utils.read_write_dense import read_array
from file_utils import read_raw_colmap_trajectory, read_gt_trajectory, \
    read_str_file, read_timestamps, align_timestamps


####################################### Parse Args ################################################
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root directory")
parser.add_argument("--split", type=str, required=True, choices=["train", "validation", "test"],
    help="Dataset split, one of [train|validation|test]")
parser.add_argument("--log_id", type=str, default=None, help="Log id")
parser.add_argument("--snippet_id", type=str, default=None, help="Snippet id for the video snippet")
parser.add_argument("--render_gt", action="store_true", default=False,
    help="Render the images with GT poses")
parser.add_argument("--mesh_fpath", type=str, default=None,
    help="Path to the 3D mesh. If none is specified, we will use a default value.")
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
        if not os.path.isdir(colmap_dir):
            print(f"{colmap_dir} is not a directory, skipping...")
            continue

        # depth_dir = os.path.join(colmap_dir, "dense", "stereo", "depth_maps")
        if args.mesh_fpath is not None and os.path.isfile(args.mesh_fpath):
            mesh_fpath = args.mesh_fpath
        else:
            mesh_fpath = os.path.join(colmap_dir, "dense", "meshed-poisson.ply")
            print(f"Using the default mesh fpath {mesh_fpath}")
        if not os.path.isfile(mesh_fpath):
            print(f"Can't find {mesh_fpath}, skipping...")
            continue

        output_dir = os.path.join(colmap_dir, "dense", "poisson_pyrender" if not args.render_gt else "poisson_pyrender_gt")
        os.makedirs(output_dir, exist_ok=True)
        if len(os.listdir(output_dir)) > 20:
            print("Output dir already has >= 20 images, continue...")
            continue

        ################################## Load Trajectories ##############################################
        print("Loading estimated trajectory...")
        traj_est = read_raw_colmap_trajectory(
            os.path.join(colmap_dir, "refined", "images.txt"),
            os.path.join(colmap_dir, "times.txt"),
            os.path.join(colmap_dir, "image_names.txt"))
        image_names_all = read_str_file(os.path.join(colmap_dir, "image_names.txt"))

        if args.render_gt:
            print("Loading GT trajectory...")
            traj_gt = read_gt_trajectory(os.path.join(args.dataset_root, args.split, f"{log_id}.txt"))
            # Associate and align trajectory
            etol = min(1000, max(0, (traj_est.timestamps[1] - traj_est.timestamps[0]) / 2))  # microseconds
            traj_gt, traj_est_aligned = associate_trajectories(traj_gt, traj_est, max_diff=etol)
            # we need to align the gt traj to est traj to match with the scale of the estimated mesh
            traj_gt.transform(np.linalg.inv(traj_gt.poses_se3[0]))
            traj_gt.align(traj_est_aligned, correct_only_scale=True)
            traj_gt.transform(traj_est.poses_se3[0])
            traj = traj_gt
            frame_ids = align_timestamps(traj_est.timestamps, traj_est_aligned.timestamps, 1)
            image_names = [image_names_all[idx] for idx in frame_ids]
        else:
            traj = traj_est
            image_names = image_names_all


        ##################################### Render ######################################################
        # read camera parameters
        with open(os.path.join(colmap_dir, "dense", "sparse", "cameras.txt"), "r") as f:
            content = f.readlines()[1].rstrip().split(" ")
            assert(len(content) == 8)
            width, height, fx, fy, cx, cy = list(map(float, content[2:]))

        # load the mesh and create pyrender scene
        tm = trimesh.load(mesh_fpath)
        mesh = pyrender.Mesh.from_trimesh(tm)

        scene = pyrender.Scene(ambient_light=[0.9, 0.9, 0.9], bg_color=[1.0, 1.0, 1.0])
        cam = pyrender.camera.IntrinsicsCamera(fx, fy, cx, cy)
        scene.add(mesh, pose=np.eye(4))
        nc = pyrender.Node(camera=cam, matrix=np.eye(4))
        scene.add_node(nc)

        r = pyrender.OffscreenRenderer(
            viewport_width=width,
            viewport_height=height,
            point_size=1.0)

        print("Start rendering...")
        for pose, img_fname in zip(traj.poses_se3, image_names):
            pyrender_pose = np.copy(pose)
            # flip the y and z axis directions for pyrender
            pyrender_pose[:3, 1] *= -1
            pyrender_pose[:3, 2] *= -1
            scene.set_pose(nc, pose=pyrender_pose)
            color, depth = r.render(scene)
            color_bgr = color[:, :, [2,1,0]]  # RGB to BGR
            cv2.imwrite(os.path.join(output_dir, img_fname+"_rendered_color.png"), color_bgr)
            # min_depth, max_depth = np.percentile(
            #     depth, [5, 95])
            # depth[depth < min_depth] = min_depth
            # depth[depth > max_depth] = max_depth
            # depth_normalized = (depth - np.min(depth))/(np.max(depth) - np.min(depth)) * 255.0
            # cv2.imwrite(os.path.join(output_dir, img_fname+"_rendered_depth.png"), depth_normalized)

        r.delete()
        print("Done!")