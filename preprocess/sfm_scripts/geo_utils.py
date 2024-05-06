import cv2
# import open3d as o3d
import numpy as np
from matplotlib import cm


# def get_camera_wireframe(scale=1.0, color=[1, 0, 0], hw_ratio=540.0/960.0, frustum_height_ratio=2.0):
#     """Make a open3d line set object that represents a camera wireframe in the camera frame."""
#     # open3d has x to the right, y up, and z backward
#     points = [
#         [-1, -hw_ratio, frustum_height_ratio/2],
#         [-1, hw_ratio, frustum_height_ratio/2],
#         [1, -hw_ratio, frustum_height_ratio/2],
#         [1, hw_ratio, frustum_height_ratio/2],
#         [0, 0, -frustum_height_ratio/2],
#     ]
#     points = np.array(points) * scale
#     lines = [
#         [0, 1],
#         [0, 2],
#         [1, 3],
#         [2, 3],
#         [0, 4],
#         [1, 4],
#         [2, 4],
#         [3, 4],
#     ]
#     line_set = o3d.geometry.LineSet()
#     line_set.points = o3d.utility.Vector3dVector(points)
#     line_set.lines = o3d.utility.Vector2iVector(lines)
#     line_set.paint_uniform_color(color)
#     return line_set


# def draw_camera_poses(camera_poses, colors=[[0.0, 0.0, 1.0]], scale=1.0, flip_yz=False):
#     """Make a series of camera wire frames with the given camera poses and colors.
#     Input:
#         camera_poses: [N x 4 x 4] list of 4x4 pose matrices from camera frame to world frame
#         colors: [N x 3] or [1 x 3]. list of rgb colors for each camera, or a uniform color
#         scale: scale of the 3D camera wireframe
#         flip_yz: flip both the y and z axis directions
#     Output:
#         a list of Open3D geometries as input to o3d.visualization.draw_geometries
#     """
#     assert(len(colors) == 1 or len(camera_poses) == len(colors))
#     if len(colors) == 1:
#         assert(len(colors[0]) == 3)
#         colors = colors * len(camera_poses)
#     camera_wireframes = []
#     for pose, color in zip(camera_poses, colors):
#         frame = get_camera_wireframe(scale=scale, color=color)
#         if flip_yz:
#             flipped_pose = np.copy(pose)
#             # flipped_pose[:3, 1] *= -1
#             # flipped_pose[:3, 2] *= -1
#             flipped_pose[1, :] *= -1
#             flipped_pose[2, :] *= -1
#             frame.transform(flipped_pose)
#         else:
#             frame.transform(pose)
#         camera_wireframes.append(frame)
#     return camera_wireframes


def get_colors(num, cmap_name):
    cmap = cm.get_cmap(name=cmap_name)
    return [cmap(x)[:3] for x in np.linspace(0.25, 1.0, num=num)]


# def test_draw_poses():
#     camera_poses = []
#     for i in range(5):
#         pose = np.eye(4)
#         pose[0, 3] = i
#         pose[2, 3] = i
#         camera_poses.append(pose)
#     primitives = draw_camera_poses(camera_poses, get_colors(len(camera_poses), "Blues"), scale=0.5, flip_yz=True)
#     o3d.visualization.draw_geometries(primitives)


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


if __name__ == "__main__":
    test_draw_poses()