import trimesh
import pyrender
import open3d as o3d
import numpy as np
import os
import argparse
import cv2
import matplotlib.pyplot as plt
from evo.core.sync import associate_trajectories

from file_utils import read_reality_capture_trajectory, read_reality_capture_trajectory_from_proj_mat, read_gt_trajectory, \
    read_str_file, read_timestamps, align_timestamps


def render(mesh, traj, intrinsics, img_names, output_base, width, height, render_depth=False):
    scene = pyrender.Scene(ambient_light=[0.9, 0.9, 0.9], bg_color=[1.0, 1.0, 1.0])
    scene.add(mesh, pose=np.eye(4))

    r = pyrender.OffscreenRenderer(
        viewport_width=width,
        viewport_height=height,
        point_size=1.0)

    for idx, (pose, K, img_fname) in enumerate(zip(traj.poses_se3, intrinsics, img_names)):
        cam = pyrender.camera.IntrinsicsCamera(K[0,0], K[1,1], K[0,2], K[1,2])
        nc = pyrender.Node(camera=cam, matrix=np.eye(4))
        scene.add_node(nc)

        pyrender_pose = np.copy(pose)
        # flip the y and z axis directions for pyrender
        pyrender_pose[:3, 1] *= -1
        pyrender_pose[:3, 2] *= -1
        scene.set_pose(nc, pose=pyrender_pose)
        color, depth = r.render(scene)

        scene.remove_node(nc)

        color_bgr = color[:, :, [2,1,0]]  # RGB to BGR
        cv2.imwrite(os.path.join(output_base, img_fname+"_rendered_color.png"), color_bgr)
        
        if render_depth:
            min_depth, max_depth = np.percentile(
                depth, [5, 95])
            depth[depth < min_depth] = min_depth
            depth[depth > max_depth] = max_depth
            depth_normalized = (depth - np.min(depth))/(np.max(depth) - np.min(depth)) * 255.0
            cv2.imwrite(os.path.join(output_base, img_fname+"_rendered_depth.png"), depth_normalized)
        
        if idx > 10:
            break

    r.delete()


####################################### Parse Args ################################################
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root directory")
parser.add_argument("--split", type=str, required=True, choices=["train", "validation", "test"],
    help="Dataset split, one of [train|validation|test]")
parser.add_argument("--log_id", type=str, required=True, help="Log id")
parser.add_argument("--snippet_id", type=str, default=None, help="Snippet id for the video snippet")
parser.add_argument("--mesh_fpath", type=str, default=None,
    help="Path to the 3D mesh. If none is specified, we will use a default value.")
parser.add_argument("--convert_to_colmap", action="store_true", default=False, help="Whether to convert to colmap format")
args = parser.parse_args()

if args.snippet_id is None:
    all_snippet_ids = os.listdir(os.path.join(args.dataset_root, "snippet", args.split, args.log_id))
else:
    all_snippet_ids = [args.snippet_id]

for snippet_id in all_snippet_ids:
    print(f"Processing snippet {snippet_id}")
    
    snippet_dir = os.path.join(args.dataset_root, "snippet", args.split, args.log_id, snippet_id)
    colmap_dir = os.path.join(snippet_dir, "colmap_output")
    reality_dir = os.path.join(snippet_dir, "reality_capture_output")

    if not os.path.isdir(reality_dir):
        print(f"{reality_dir} is not a directory, skipping...")
        continue

    # depth_dir = os.path.join(colmap_dir, "dense", "stereo", "depth_maps")
    if args.mesh_fpath is not None and os.path.isfile(args.mesh_fpath):
        mesh_fpath = args.mesh_fpath
    else:
        mesh_fpath = os.path.join(reality_dir, "recon.ply")
        print(f"Using the default mesh fpath {mesh_fpath}")
    
    if not os.path.isfile(mesh_fpath):
        print(f"Can't find {mesh_fpath}, skipping...")
        continue

    output_dir = os.path.join(reality_dir, "pyrender")
    os.makedirs(output_dir, exist_ok=True)

    ################################## Load Trajectories ##############################################
    print("Loading estimated trajectory...")
    traj_euler, intrinsics_euler, traj_img_names_euler = read_reality_capture_trajectory(
        os.path.join(snippet_dir, "reality_capture_output", "poses.csv"),
        os.path.join(snippet_dir, "colmap_output", "times.txt"),
        os.path.join(snippet_dir, "colmap_output", "image_names.txt"),
        os.path.join(snippet_dir, "reality_capture_output", "P_matrix"),
        # os.path.join(snippet_dir, "colmap_output", "dense", "images"),
        convert_to_colmap=args.convert_to_colmap,
        transpose=True)

    if os.path.exists(os.path.join(snippet_dir, "reality_capture_output", "P_matrix")):
        traj_proj, intrinsics_proj, traj_img_names_proj, _ = read_reality_capture_trajectory_from_proj_mat(
            os.path.join(snippet_dir, "reality_capture_output", "P_matrix"),
            os.path.join(snippet_dir, "colmap_output", "times.txt"),
            os.path.join(snippet_dir, "colmap_output", "image_names.txt"),
            convert_to_colmap=args.convert_to_colmap,
            transpose=True)
    else:
        print("Proj mat folder doesn't exist, skipping...")
        traj_proj = None
    print("Trajectories loaded!")

    # from IPython import embed
    # embed()
    # assert False

    ##################################### Render ######################################################
    # read camera parameters
    raw_img = cv2.imread(os.path.join(snippet_dir, "reality_capture_output", "P_matrix", traj_img_names_proj[0]))
    width, height = raw_img.shape[0], raw_img.shape[1]  # we transpose them


    tm = trimesh.load(mesh_fpath)
    if args.convert_to_colmap:
        colmap_transform = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        tm.vertices = (np.matmul(colmap_transform[:3, :3], tm.vertices.T) + colmap_transform[:3, 3][:, None]).T

    mesh = pyrender.Mesh.from_trimesh(tm)

    euler_stem = "euler" if not args.convert_to_colmap else "euler_colmap_format"
    os.makedirs(os.path.join(output_dir, euler_stem), exist_ok=True)
    print(f"Rendering for {euler_stem}")
    render(mesh, traj_euler, intrinsics_euler, traj_img_names_euler, os.path.join(output_dir, euler_stem), width, height)
    if traj_proj is not None:
        projmat_stem = "proj_mat" if not args.convert_to_colmap else "proj_mat_colmap_format"
        os.makedirs(os.path.join(output_dir, projmat_stem), exist_ok=True)
        print(f"Rendering for {projmat_stem}")
        render(mesh, traj_proj, intrinsics_proj, traj_img_names_proj, os.path.join(output_dir, projmat_stem), width, height)