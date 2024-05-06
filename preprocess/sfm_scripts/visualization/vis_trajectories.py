import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import math

import seaborn as sns
sns.set()
sns.set_style("whitegrid")

from evo.core.trajectory import PoseTrajectory3D
from evo.core.sync import associate_trajectories

from file_utils import read_gt_trajectory, read_tum_trajectory, read_raw_colmap_trajectory, \
    read_raw_colmap_sparse_pointcloud, read_reality_capture_trajectory, read_reality_capture_trajectory_from_proj_mat
from geo_utils import draw_camera_poses, get_colors


####################################### Parse Args ################################################
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root directory")
parser.add_argument("--split", type=str, required=True, choices=["train", "validation", "test"],
    help="Dataset split, one of [train|validation|test]")
parser.add_argument("--log_id", type=str, required=True, help="Log id")
parser.add_argument("--snippet_id", type=str, required=True, help="Snippet id for the video snippet")
parser.add_argument("--traj_type", type=str, choices=["orb", "colmap", "reality"], required=True,
    help="Trajectory type. Choices: [orb | colmap | reality]")
parser.add_argument("--mesh_fpath", type=str, default=None,
    help="Path to a 3D mesh. If none is specified, the mesh won't be visualized. You can use 'fuse', "
    "'sparse' as shorthand to load the dense or sparse point clouds from the default file structure.")
parser.add_argument("--mesh_type", type=str, default="triangle", choices=["triangle", "pointcloud"], 
    help="Type of the mesh, one of [triangle|pointcloud], default is triangle mesh.")
parser.add_argument("--compare_with", type=str, choices=["orb", "colmap", "reality", "gt"], default=None,
    help="Trajectory to compare")
args = parser.parse_args()

snippet_dir = os.path.join(args.dataset_root, "snippet", args.split, args.log_id, args.snippet_id)
assert(os.path.isdir(snippet_dir))
output_dir = os.path.join(snippet_dir, "visualization")
os.makedirs(output_dir, exist_ok=True)


################################## Load Trajectories ##############################################
def proj_mesh(pts, pose=None, K=None, proj_mat=None, fpath=None):
    x = np.array(pts).T
    if K is not None and pose is not None:
        extrinsics = np.linalg.inv(pose)
        R, t = extrinsics[:3, :3], extrinsics[:3, 3][:, None]
        x_im = K.dot(R.dot(x) + t)
    else:
        assert proj_mat is not None
        x_im = proj_mat.dot(np.concatenate((x, np.ones((1, x.shape[1])))))
    z = x_im[2, :]
    x = x_im[:2, :] / (x_im[2, :]+1e-6)
    plt.figure()
    # plt.imshow(im_undist)
    # plt.figure()
    # idx = np.random.permutation(x.shape[1])[:int(x.shape[1]/2)]
    plt.scatter(x[0], x[1], c = np.clip(z, 0, 100.0), s = 0.1)
    # plt.scatter(x[0, idx], x[1, idx], c = np.clip(z[idx], 0, 100.0), s = 0.1)
    plt.xlim([0, 1600])
    plt.ylim([900, 0])
    ax = plt.gca()
    ax.set_aspect('equal')
    if fpath is not None:
        plt.savefig(fpath)
    else:
        plt.show()
    plt.close()


def load_traj(traj_type):
    if traj_type == "orb":
        traj = read_tum_trajectory(os.path.join(snippet_dir, "orb_output", "KeyFrameTrajectory.txt"))
        traj_proj = None
    elif traj_type == "colmap":
        traj = read_raw_colmap_trajectory(
            os.path.join(snippet_dir, "colmap_output", "refined", "images.txt"),
            os.path.join(snippet_dir, "times.txt"),
            # os.path.join(snippet_dir, "colmap_output", "times.txt"),
            os.path.join(snippet_dir, "image_names.txt"))
            # os.path.join(snippet_dir, "colmap_output", "image_names.txt"))
        traj_proj = None
    elif traj_type == "reality":
        traj, intrinsics, _ = read_reality_capture_trajectory(
            os.path.join(snippet_dir, "reality_capture_output", "poses.csv"),
            os.path.join(snippet_dir, "colmap_output", "times.txt"),
            os.path.join(snippet_dir, "colmap_output", "image_names.txt"),
            os.path.join(snippet_dir, "reality_capture_output", "P_matrix"), transpose=True)
            # os.path.join(snippet_dir, "colmap_output", "dense", "images"))
        traj_proj, intrinsics_proj, _, proj_mats = read_reality_capture_trajectory_from_proj_mat(
            os.path.join(snippet_dir, "reality_capture_output", "P_matrix"),
            os.path.join(snippet_dir, "colmap_output", "times.txt"),
            os.path.join(snippet_dir, "colmap_output", "image_names.txt"), transpose=True)
        
        from IPython import embed
        embed()
        
        # num_vis = 2
        # pts = o3d.io.read_point_cloud(os.path.join(snippet_dir, "reality_capture_output", "recon.ply"))
        # for idx, (pose, K, pose_proj, K_proj, proj_mat) in enumerate(zip(
        #     traj.poses_se3[:num_vis], intrinsics[:num_vis],
        #     traj_proj.poses_se3[:num_vis], intrinsics_proj[:num_vis], proj_mats[:num_vis]
        # )):
        #     proj_mesh(np.copy(np.asarray(pts.points)), pose=pose, K=K, fpath=os.path.join(snippet_dir, "reality_capture_output", f"rendered_img_{idx:03d}.png"))
        #     proj_mesh(np.copy(np.asarray(pts.points)), pose=pose_proj, K=K_proj, fpath=os.path.join(snippet_dir, "reality_capture_output", f"rendered_img_{idx:03d}_proj.png"))
        #     proj_mesh(np.copy(np.asarray(pts.points)), proj_mat=proj_mat, fpath=os.path.join(snippet_dir, "reality_capture_output", f"rendered_img_{idx:03d}_proj_mat.png"))
        # from IPython import embed
        # embed()
        # assert False


    elif traj_type == "gt":
        traj = read_gt_trajectory(os.path.join(args.dataset_root, args.split, f"{args.log_id}.txt"))
        traj_proj = None
    else:
        raise ValueError(f"Unknown traj type {args.traj_type}")
    return traj
    # return traj_proj if traj_proj is not None else traj


print("Loading estimated trajectory...")
traj_est = load_traj(args.traj_type)
if args.compare_with is not None:
    print("Loading trajectory to compare...")
    traj_comp = load_traj(args.compare_with)
    etol = min(1000, max(0, (traj_est.timestamps[1] - traj_est.timestamps[0]) / 2))  # microseconds
    traj_comp, traj_est = associate_trajectories(traj_comp, traj_est, max_diff=etol)
    if args.mesh_fpath is None:
        # we scale the est traj according to the comp traj
        traj_comp.transform(np.linalg.inv(traj_comp.poses_se3[0]))
        traj_est.align_origin(traj_comp)
        traj_est.align(traj_comp, correct_only_scale=True)
    else:
        # we need to align the gt traj to est traj to match with the scale of the estimated mesh
        traj_comp.transform(np.linalg.inv(traj_comp.poses_se3[0]))
        traj_comp.align(traj_est, correct_only_scale=True)
        traj_comp.transform(traj_est.poses_se3[0])
else:
    traj_comp = None


################################## Visualization ##################################################
def visualize_trajectory_2d(traj_comp, traj_est):
    """Plot the trajectories in 2D BEV."""
    ax = plt.gca()
    if traj_comp is not None:
        ax.plot([pose[0, 3] for pose in traj_comp.poses_se3], [pose[2, 3] for pose in traj_comp.poses_se3], label=args.compare_with.upper())
    ax.plot([pose[0, 3] for pose in traj_est.poses_se3], [pose[2, 3] for pose in traj_est.poses_se3], label=args.traj_type.upper())
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()
    # suffix = args.traj_type
    # plt.savefig(os.path.join(output_dir, f"{log_id}_traj_{suffix}.png"))
    # plt.close()


def visualize_trajectory_3d(traj_comp, traj_est):
    """Plot the trajectories in 3D."""
    avg_travel_dist = np.mean([
        np.linalg.norm(np.matmul(np.linalg.inv(traj_est.poses_se3[i]), traj_est.poses_se3[i+1])[:3, 3]) \
            for i in range(len(traj_est.poses_se3)-1)
    ])
    scale = max(1e-4, avg_travel_dist/2)  # heuristics to auto adjust the scale of the camera wireframes
    if traj_comp is not None:
        traj_comp_frames = draw_camera_poses(
            traj_comp.poses_se3,
            get_colors(len(traj_comp.poses_se3), "Blues"),
            scale=scale, flip_yz=True)
    else:
        traj_comp_frames = []
    traj_est_frames = draw_camera_poses(
        traj_est.poses_se3,
        get_colors(len(traj_est.poses_se3), "Oranges"),
        scale=scale, flip_yz=True)
    mesh = []
    if args.mesh_fpath is not None:
        mesh_type = args.mesh_type
        if not os.path.isfile(args.mesh_fpath):
            if "fuse" in args.mesh_fpath:
                # use a default point cloud mesh
                mesh_fpath = os.path.join(snippet_dir, "colmap_output", "dense", "fused.ply")
                mesh_type = "pointcloud"
            elif "sparse" in args.mesh_fpath:
                # use a default path to the sparse 3D point cloud
                mesh_fpath = os.path.join(snippet_dir, "colmap_output", "refined", "points3D.txt")
                mesh_type = "pointcloud"
            elif "reality" in args.mesh_fpath:
                # use a default path to the reality capture point cloud
                mesh_fpath = os.path.join(snippet_dir, "reality_capture_output", "recon.ply")
                mesh_type = "triangle"
            else:
                # use a default mesh
                mesh_fpath = os.path.join(snippet_dir, "colmap_output", "dense", "meshed-poisson.ply")
                mesh_type = "triangle"
            print(f"Input mesh fpath {args.mesh_fpath} does not exist. Use default {mesh_fpath} instead.")
        else:
            mesh_fpath = args.mesh_fpath
        if os.path.isfile(mesh_fpath):
            if mesh_type == "pointcloud":
                if mesh_fpath.endswith("txt"):
                    xyz, rgb = read_raw_colmap_sparse_pointcloud(mesh_fpath)
                    mesh = o3d.geometry.PointCloud()
                    mesh.points = o3d.utility.Vector3dVector(xyz)
                    mesh.colors = o3d.utility.Vector3dVector(rgb)
                else:
                    mesh = o3d.io.read_point_cloud(mesh_fpath)
                if "colmap" in mesh_fpath:
                    # flip the y and z directions, since colmap and open3d have opposite conventions
                    np.asarray(mesh.points)[:, 1] *= -1
                    np.asarray(mesh.points)[:, 2] *= -1
                elif "reality" in mesh_fpath:
                    # transpose the y and z directions
                    np.asarray(mesh.points)[:, [1,2]] = np.asarray(mesh.points)[:, [2,1]]
                    np.asarray(mesh.points)[:, 1] *= -1
                # clip the depths
                # min_depth, max_depth = np.percentile(np.asarray(mesh.points)[:, 2], [5, 95])
                # np.asarray(mesh.points)[np.asarray(mesh.points)[:, 2] < min_depth] = min_depth
                # np.asarray(mesh.points)[np.asarray(mesh.points)[:, 2] > max_depth] = max_depth
            else:
                mesh = o3d.io.read_triangle_mesh(mesh_fpath)
                if "colmap" in mesh_fpath:
                    # flip the y and z directions, since colmap and open3d have opposite conventions
                    np.asarray(mesh.vertices)[:, 1] *= -1
                    np.asarray(mesh.vertices)[:, 2] *= -1
                elif "reality" in mesh_fpath:
                    np.asarray(mesh.vertices)[:, [1,2]] = np.asarray(mesh.vertices)[:, [2,1]]
                    np.asarray(mesh.vertices)[:, [0,2]] = np.asarray(mesh.vertices)[:, [2,0]]
                    np.asarray(mesh.vertices)[:, 1] *= -1
                    np.asarray(mesh.vertices)[:, 2] *= -1
                    # pass
                    # transpose the y and z directions
                    # np.asarray(mesh.vertices)[:, [1,2]] = np.asarray(mesh.vertices)[:, [2,1]]
                    # np.asarray(mesh.vertices)[:, 2] *= -1
            mesh = [mesh]
    o3d.visualization.draw_geometries(traj_comp_frames + traj_est_frames + mesh)


visualize_trajectory_2d(traj_comp, traj_est)
visualize_trajectory_3d(traj_comp, traj_est)