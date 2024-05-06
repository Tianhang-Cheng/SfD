
"""Convert ORB-SLAM output into COLMAP formats. Prepare images.txt, camera.txt and points3D.txt"""
import cv2
import os
import numpy as np
import argparse

import open3d as o3d
from evo.core.trajectory import PoseTrajectory3D

from file_utils import get_intrinsics, read_timestamps, read_str_file, read_tum_trajectory, \
    align_timestamps, get_colmap_image_name_to_id_mapping, read_raw_colmap_camera_multiple, read_raw_colmap_trajectory
from geo_utils import rotmat2qvec
from colmap_utils.read_write_dense import read_array


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

################################# Load estimated COLMAP trajectory ###################################
colmap_imgs = sorted([img_name for img_name in os.listdir(os.path.join(snippet_dir, "reality_capture_output", "P_matrix")) if img_name.endswith(".png")])
traj = read_raw_colmap_trajectory(os.path.join(colmap_dir, "dense", "sparse", "images.txt"), list(range(len(colmap_imgs))), colmap_imgs)
cam_intrs = read_raw_colmap_camera_multiple(os.path.join(colmap_dir, "dense", "sparse", "cameras.txt"))
assert len(cam_intrs) == len(traj.poses_se3)


volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=50.0 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

img_sample = cv2.imread(os.path.join(colmap_dir, "dense", "images", colmap_imgs[0]))
width = img_sample.shape[1]
height = img_sample.shape[0]
points = []
for i, img_name in enumerate(colmap_imgs):
    print("Integrate {:d}-th image into the volume.".format(i))
    color = o3d.io.read_image(os.path.join(colmap_dir, "dense", "images", img_name))
    depth_fpath = os.path.join(colmap_dir, "dense", "stereo", "depth_maps", f"{img_name}.geometric.png")
    # if not os.path.exists(depth_fpath):
    #     depth_map = read_array(os.path.join(colmap_dir, "dense", "stereo", "depth_maps", f"{img_name}.geometric.bin"))
    #     min_depth, max_depth = np.percentile(
    #         depth_map, [5, 95])
    #     depth_map[depth_map < min_depth] = min_depth
    #     depth_map[depth_map > max_depth] = max_depth
    #     cv2.imwrite(depth_fpath, depth_map / 1000.0)

    # depth = o3d.io.read_image(depth_fpath)
    depth_map = np.ascontiguousarray(read_array(os.path.join(colmap_dir, "dense", "stereo", "depth_maps", f"{img_name}.geometric.bin")))
    min_depth, max_depth = np.percentile(
        depth_map, [5, 95])
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth
    depth = o3d.geometry.Image((depth_map / max_depth * 255).astype(np.uint8))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=1.0/max_depth, depth_trunc=150.0, convert_rgb_to_intensity=False)
    K = o3d.camera.PinholeCameraIntrinsic()
    K.set_intrinsics(width, height, cam_intrs[i][0,0], cam_intrs[i][1,1], cam_intrs[i][0,2], cam_intrs[i][1,2])
    volume.integrate(
        rgbd,
        K,
        np.linalg.inv(traj.poses_se3[i])
    )
    # o3d.visualization.draw_geometries([rgbd])
    # from IPython import embed
    # embed()
    # assert False
    # depth_map = read_array(os.path.join(colmap_dir, "dense", "stereo", "depth_maps", f"{img_name}.geometric.bin"))
    # min_depth, max_depth = np.percentile(
    #     depth_map, [5, 95])
    # depth_map[depth_map < min_depth] = min_depth
    # depth_map[depth_map > max_depth] = max_depth
    # uvd = np.ones([width * height, 3])
    # uvd[:, 0] = np.tile(np.arange(width), height)
    # uvd[:, 1] = np.repeat(np.arange(height), width)
    # # from IPython import embed
    # # embed()
    # # assert False
    # # points3d = np.matmul(np.linalg.inv(cam_intrs[i]), uvd.T)  # 3 x P
    # points3d = uvd.T
    # points3d[0, :] = (points3d[0, :] - cam_intrs[i][0, 2]) / cam_intrs[i][0, 0]
    # points3d[1, :] = (points3d[1, :] - cam_intrs[i][1, 2]) / cam_intrs[i][1, 1]
    # points3d *= depth_map.ravel()[None, :]
    # points3d = points3d[:, points3d[2, :] > 0]
    # points3d = points3d[:, points3d[2, :] < 200]

    # points3d_homo = np.vstack([points3d, np.ones([1, points3d.shape[1]])]) # 4 x P
    # points3d_homo = np.matmul(traj.poses_se3[i], points3d_homo) # 4 x P
    # points3d_homo = points3d_homo.T
    # points3d = points3d_homo[:, :3] / points3d_homo[:, [3]]
    # points.append(points3d)
    # points_o3d = o3d.geometry.PointCloud()
    # points_o3d.points = o3d.utility.Vector3dVector(np.concatenate(points))
    # o3d.visualization.draw_geometries([points_o3d])

# mesh = volume.extract_triangle_mesh()
# mesh = volume.extract_point_cloud()
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh])
from IPython import embed
embed()
