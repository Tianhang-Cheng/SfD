import numpy as np
import os
import smplx
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_style("whitegrid")

from baselines.pyceres import ceres_ba

np.random.seed(1000)


def project(points, transformation, K):
    points_homo = np.hstack((points, np.ones([points.shape[0], 1])))
    points_transformed = np.matmul(transformation, points_homo.T)
    points_transformed = points_transformed[:3] / points_transformed[3]
    points_uvd = np.matmul(K, points_transformed)
    points_uvd[:2] /= points_uvd[2]
    return points_uvd.T


def dump_projection(points, transformation, K):
    points_uv = project(points, transformation, K)[:, :2]
    out_dir = os.path.join(os.path.expanduser(os.path.expandvars("~/generalized_sfm/smpl/")), "vis")
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for pt in points_uv:
        ax.scatter(pt[0], pt[1])
    fig.savefig(os.path.join(out_dir, "projection.png"))
    plt.close()


model_fpath = os.path.expanduser(os.path.expandvars("~/generalized_sfm/smpl/data/SMPL_NEUTRAL.pkl"))
smpl_model = smplx.create(model_path=model_fpath)

output = smpl_model(return_verts=True)
joints = output.joints.detach().cpu().numpy().squeeze()

img_width = 960
img_height = 540
cam_intrinsics = np.array([
    [906.66786624, 0.0, 480.0],
    [0.0, 913.2243344999999, 270.0],
    [0.0, 0.0, 1.0]
])

gt_pose = np.eye(4)
gt_pose[2, 3] = -5
gt_ext = np.linalg.inv(gt_pose)

project_uvd = project(joints, gt_ext, cam_intrinsics)
dump_projection(joints, gt_ext, cam_intrinsics)
# is_in_img = (project_uv[:, 0] >= 0) & (project_uv[:, 0] < img_width) & (project_uv[:, 1] >= 0) & (project_uv[:, 1] < img_height)

observations = []
points = []
for pt_2d, pt_3d in zip(project_uvd, joints):
    u, v, d = pt_2d
    if (u >= 0) and (u < img_width) and (v >= 0) and (v < img_height) and (d > 0):
        observations.append([float(u), float(v), len(points)])
        points.append(pt_3d.tolist())

est_ext, points_ba = ceres_ba(observations, cam_intrinsics.tolist(), points, gt_ext.tolist())
est_ext = np.array(est_ext)
print(est_ext)
assert(np.allclose(gt_ext, est_ext))

pose_init = np.eye(4)
pose_init[2, 3] = -3
est_ext, points_ba = ceres_ba(observations, cam_intrinsics.tolist(), points, np.linalg.inv(pose_init).tolist())
est_ext = np.array(est_ext)
print(est_ext)
# assert(np.allclose(gt_ext, est_ext))

noisy_points = (np.array(points) + np.random.rand(*(np.array(points).shape)) * 1e-4).tolist()
pose_init = np.eye(4)
pose_init[2, 3] = -4
est_ext, points_ba = ceres_ba(observations, cam_intrinsics.tolist(), noisy_points, np.linalg.inv(pose_init).tolist())
est_ext = np.array(est_ext)
print(est_ext)
print(np.linalg.norm(est_ext - gt_ext))
print(np.linalg.norm(np.array(points_ba) - points))
from IPython import embed
embed()