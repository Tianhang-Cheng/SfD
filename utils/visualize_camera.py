import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_save_poses_simple(pose, pose_ref=None, points=None, path='pose.png',
                            scale=1, cam_depth_scale=3, show=False,
                            camera_look_at_positive_z=False):
    """
    Minimal camera pose visualizer: plots camera centers (and optionally a
    reference set of poses / a point cloud) in 3D and saves the figure.

    pose, pose_ref: [n, 4, 4] camera-to-world matrices (numpy or torch)
    points: optional [m, 3] point cloud
    """
    if hasattr(pose, 'detach'):
        pose = pose.detach().cpu().numpy()
    if pose_ref is not None and hasattr(pose_ref, 'detach'):
        pose_ref = pose_ref.detach().cpu().numpy()
    if points is not None and hasattr(points, 'detach'):
        points = points.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    centers = pose[:, :3, 3]
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
               c='blue', marker='o', label='pred')

    if pose_ref is not None:
        centers_ref = pose_ref[:, :3, 3]
        ax.scatter(centers_ref[:, 0], centers_ref[:, 1], centers_ref[:, 2],
                   c='red', marker='^', label='gt')

    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c='gray', marker='.', alpha=0.3, s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Camera pose alignment')
    plt.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
