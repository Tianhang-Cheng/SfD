import numpy as np
import subprocess
# import open3d as o3d
import os
import torch
import cv2

# from evo.core.trajectory import PoseTrajectory3D
# from evo.core.sync import associate_trajectories

from sfm_scripts.file_utils import read_raw_colmap_sparse_pointcloud
from sfm_scripts.geo_utils import qvec2rotmat, get_colors, rotmat2qvec # draw_camera_poses
 
def estimate_relative_pose(kpts0, kpts1, K0, K1, thresh=1, conf=0.99999):
    if len(kpts0) < 5:
        return None, None
    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)
    assert E is not None
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret, E


def run_geometric_verification(db_fpath, match_list_fpath, colmap_path="", hide_output=False, type="pairs"):
    if hide_output:
        pipe = subprocess.DEVNULL
    else:
        pipe = None
    subprocess.call([os.path.join(colmap_path, "colmap"), "matches_importer",
                     "--database_path", db_fpath,
                     "--match_list_path", match_list_fpath,
                     "--match_type", type],
                     stdout=pipe, stderr=pipe)


def run_mapper(db_fpath, img_path, out_path, colmap_path="", hide_output=False):
    if hide_output:
        pipe = subprocess.DEVNULL
    else:
        pipe = None
    subprocess.call([os.path.join(colmap_path, "colmap"), "mapper",
                     "--database_path", db_fpath,
                     "--image_path", img_path,
                     "--output_path", out_path,
                    #  "--Mapper.tri_merge_max_reproj_error", "3",
                    #  "--Mapper.filter_max_reproj_error", "2",
                     "--Mapper.init_min_num_inliers", "5",
                     "--Mapper.min_num_matches", "5",
                     "--Mapper.abs_pose_min_num_inliers", "5",
                     "--Mapper.abs_pose_min_inlier_ratio", "0.1",
                     "--Mapper.ba_refine_focal_length", "0",
                     "--Mapper.ba_refine_principal_point", "0",
                     "--Mapper.ba_refine_extra_params", "0"],
                     stdout=pipe, stderr=pipe)


def run_triag_ba(db_fpath, img_path, input_path, sparse_out_path, ba_out_path, colmap_path="", hide_output=False):
    if hide_output:
        pipe = subprocess.DEVNULL
    else:
        pipe = None
    os.makedirs(sparse_out_path, exist_ok=True)
    os.makedirs(ba_out_path, exist_ok=True)
    subprocess.call([os.path.join(colmap_path, "colmap"), "point_triangulator",
                    "--database_path", db_fpath,
                    "--image_path", img_path,
                    "--input_path", input_path,
                    "--output_path", sparse_out_path,
                    "--Mapper.min_num_matches", "5",
                    # "--Mapper.tri_merge_max_reproj_error", "3",
                    "--Mapper.filter_max_reproj_error", "10",
                    "--Mapper.tri_re_min_ratio", "0.1",
                    "--Mapper.tri_complete_max_reproj_error", "10",
                    "--Mapper.init_min_num_inliers", "5",
                    "--Mapper.abs_pose_min_num_inliers", "5",
                    "--Mapper.abs_pose_min_inlier_ratio", "0.1",
                    "--Mapper.ba_refine_focal_length", "0",
                    "--Mapper.ba_refine_principal_point", "0",
                    "--Mapper.ba_refine_extra_params", "0"],
                    stdout=pipe, stderr=pipe)
    subprocess.call([os.path.join(colmap_path, "colmap"), "bundle_adjuster",
                    "--input_path", sparse_out_path,
                    "--output_path", ba_out_path,
                    "--BundleAdjustment.refine_focal_length", "0",
                    "--BundleAdjustment.refine_principal_point", "0",
                    "--BundleAdjustment.refine_extra_params", "0"],
                    stdout=pipe, stderr=pipe)


def gen_triag_input(colmap_input_dir, camera_info, img_exts):
    os.makedirs(colmap_input_dir)

    # cameras.txt
    with open(os.path.join(colmap_input_dir, "cameras.txt"), "w") as f:
        for camera_id, width, height, K in camera_info:
            f.write("{} PINHOLE {} {} {} {} {} {}\n".format(camera_id, width * 1.0, height * 1.0, K[0, 0], K[1, 1], K[0, 2], K[1, 2]))

    # images.txt
    # load the database and find the image id to image name mapping
    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    # Number of images: 2, mean observations per image: 2
    # see https://colmap.github.io/format.html and https://colmap.github.io/faq.html
    with open(os.path.join(colmap_input_dir, "images.txt"), "w") as f:
        for (img_id, img_name, ext), (camera_id, _, _, _) in zip(img_exts, camera_info):
            qvec = rotmat2qvec(ext[:3, :3])
            f.write("{} {} {} {} {} {} {} {} {} {}\n".format(
                img_id, qvec[0], qvec[1], qvec[2], qvec[3],
                ext[0,3], ext[1,3], ext[2,3], camera_id, img_name))
            f.write("\n")  # empty line for keypoints

    # points_3d.txt
    # should be an empty file
    open(os.path.join(colmap_input_dir, "points3D.txt"), "a").close()


def load_from_match_list_file(fpath):
    with open(fpath, "r") as f:
        raw_matches = [line.strip().split(" ") for line in f.readlines()]
    img_names_all = sorted(list(set([img_name for pairs in raw_matches for img_name in pairs])))
    return img_names_all, raw_matches


def read_colmap_trajectory(colmap_output_fpath):
    colmap_traj_image_names = []
    poses = {}
    with open(colmap_output_fpath, "r") as f:
        lines = f.readlines()
    # first row of images.txt is a comment, poses are every other line
    # starting from the second row
    for line_idx in range(1, len(lines), 2):
        content = lines[line_idx].rstrip().split(" ")
        assert(len(content) == 10)
        colmap_traj_image_names.append(content[-1])
        pose = np.eye(4)
        qvec = list(map(float, content[1:5]))
        pose[:3, :3] = qvec2rotmat(qvec)
        pose[:3, 3] = list(map(float, content[5:8])) # pose is from world to camera frame
        poses[content[-1]] = pose
    return poses


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def from_Rt(R, t):
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t.reshape(-1)
    return pose


def compute_metrics(poses, matches_all, data_all):
    results = {}
    for img1_name, img2_name in matches_all:
        assert (img1_name, img2_name) in data_all['pair']
        data_pair = data_all['pair'][(img1_name, img2_name)]
        
        if not img1_name in poses.keys() or not img2_name in poses.keys():
            continue
        
        gt_pose1 = from_Rt(data_all['unary'][img1_name]['R'], data_all['unary'][img1_name]['t']) 
        gt_pose2 = from_Rt(data_all['unary'][img2_name]['R'], data_all['unary'][img2_name]['t'])
        est_pose1 = poses[img1_name]
        est_pose2 = poses[img2_name]
        gt_rel = np.matmul(gt_pose2, np.linalg.inv(gt_pose1))
        assert(np.allclose(gt_rel[:3, :3], data_pair['R_gt']))
        assert(np.allclose(gt_rel[:3, 3], data_pair['t_gt']))
        
        est_rel = np.matmul(est_pose2, np.linalg.inv(est_pose1))
        
        err_R = angle_error_mat(gt_rel[:3, :3], est_rel[:3, :3])
        err_t = angle_error_vec(gt_rel[:3, 3], est_rel[:3, 3])
        # print(img1_name, img2_name)
        # print("angle_err:", err_R)
        # print("angle_err_sg_F:", data_pair["ang_err"])
        # print("angle_t:", err_t)
        # print("angle_t_sg_F:", data_pair["trans_err"])
        # print()
        
        results[(img1_name, img2_name)] = {"ang_err": err_R, "trans_err": err_t}
    return results


def dump_poses(colmap_fpath, image_names, out_fpath):
    colmap_traj = read_colmap_trajectory(colmap_fpath)
    print(sorted(list(colmap_traj.keys())))
    print('\n')
    print(sorted(image_names))
    if sorted(list(colmap_traj.keys())) != sorted(image_names):
        print("Warning: not all images have an associated pose")
    np.savez(out_fpath, colmap_traj)
    return colmap_traj


# def vis_poses_3D_with_gt(est_poses_dict, img_names_all, data_all, pcd_fpath=None):
#     gt_times = []
#     gt_poses = []
#     est_times = []
#     est_poses = []
#     for idx, img_name in enumerate(img_names_all):
#         gt_times.append(float(idx))  # fake time
#         gt_poses.append(np.linalg.inv(from_Rt(data_all['unary'][img_name]['R'], data_all['unary'][img_name]['t'])))
#         if img_name in est_poses_dict.keys():
#             est_times.append(float(idx))
#             est_poses.append(np.linalg.inv(est_poses_dict[img_name]))
#     traj_gt = PoseTrajectory3D(timestamps=np.array(gt_times), poses_se3=gt_poses)
#     traj_est = PoseTrajectory3D(timestamps=np.array(est_times), poses_se3=est_poses)

#     traj_gt, traj_est = associate_trajectories(traj_gt, traj_est)
#     # if pcd_fpath is None:
#     #     # we scale the est traj according to the comp traj
#     #     traj_gt.transform(np.linalg.inv(traj_gt.poses_se3[0]))
#     #     traj_est.align_origin(traj_gt)
#     #     traj_est.align(traj_gt, correct_only_scale=True)
#     # else:
#     #     # we need to align the gt traj to est traj to match with the scale of the estimated mesh
#     #     traj_est.transform(np.linalg.inv(traj_est.poses_se3[0]))
#     #     traj_gt.align_origin(traj_est)
#     #     traj_gt.align(traj_est, correct_only_scale=True)
#     #     # traj_gt.transform(np.linalg.inv(traj_gt.poses_se3[0]))
#     #     # traj_gt.align(traj_est, correct_only_scale=True)
#     #     # traj_gt.transform(traj_est.poses_se3[0])
    
#     avg_travel_dist_gt = np.mean([
#         np.linalg.norm(np.matmul(np.linalg.inv(traj_gt.poses_se3[i]), traj_gt.poses_se3[i+1])[:3, 3]) \
#             for i in range(len(traj_gt.poses_se3)-1)
#     ])
#     scale_gt = max(1e-4, avg_travel_dist_gt/10)  # heuristics to auto adjust the scale of the camera wireframes

#     avg_travel_dist_est = np.mean([
#         np.linalg.norm(np.matmul(np.linalg.inv(traj_est.poses_se3[i]), traj_est.poses_se3[i+1])[:3, 3]) \
#             for i in range(len(traj_est.poses_se3)-1)
#     ])
#     scale_est = max(1e-4, avg_travel_dist_est/10)  # heuristics to auto adjust the scale of the camera wireframes
#     traj_gt_frames = draw_camera_poses(
#         traj_gt.poses_se3,
#         get_colors(len(traj_gt.poses_se3), "Blues"),
#         scale=scale_gt, flip_yz=True)
#     traj_est_frames = draw_camera_poses(
#         traj_est.poses_se3,
#         get_colors(len(traj_est.poses_se3), "Oranges"),
#         scale=scale_est, flip_yz=True)
#     # traj_est_frames = []
#     pcds = []
#     if pcd_fpath is not None:
#         xyz, rgb = read_raw_colmap_sparse_pointcloud(pcd_fpath)
#         pcd = o3d.geometry.PointCloud()
#         xyz[:, 1] *= -1
#         xyz[:, 2] *= -1
#         pcd.points = o3d.utility.Vector3dVector(xyz)
#         pcd.colors = o3d.utility.Vector3dVector(rgb)
#         pcds.append(pcd)
#     print(img_names_all)
#     o3d.visualization.draw_geometries(traj_est_frames + pcds)
#     o3d.visualization.draw_geometries(traj_gt_frames)

# Mutual nearest neighbors matcher for L2 normalized descriptors.
def mutual_nn_matcher(descriptors1, descriptors2):
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    matches = torch.stack([ids1[mask], nn12[mask]]).t()
    return matches.data.cpu().numpy()


# Symmetric Lowe's ratio test matcher for L2 normalized descriptors.
def ratio_matcher(descriptors1, descriptors2, ratio=0.8, symmetric=False):
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn12 = nns[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    if symmetric:
        nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
        nns_dist = torch.sqrt(2 - 2 * nns_sim)
        # Compute Lowe's ratio.
        ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
        # Save first NN.
        nn21 = nns[:, 0]
        
        # Symmetric ratio test.
        ids1 = torch.arange(0, sim.shape[0], device=device)
        mask = torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio)
    else:
        ids1 = torch.arange(0, sim.shape[0], device=device)
        mask = ratios12 <= ratio
    
    # Final matches.
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)

    return matches.data.cpu().numpy()


# Mutual NN + symmetric Lowe's ratio test matcher for L2 normalized descriptors.
def mutual_nn_ratio_matcher(descriptors1, descriptors2, ratio=0.8):
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN and match similarity.
    nn12 = nns[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn21 = nns[:, 0]
    
    # Mutual NN + symmetric ratio test.
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = torch.min(ids1 == nn21[nn12], torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio))
    
    # Final matches.
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)

    return matches.data.cpu().numpy()