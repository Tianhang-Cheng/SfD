import torch

from sklearn.cluster import KMeans
import numpy as np

def keep_points_near_center(points, num_clusters=1, keep_ratio=0.95):
    # 使用K-Means聚类算法找到点云中心
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    kmeans.fit(points)
    center = kmeans.cluster_centers_[0]

    # 计算每个点距离中心的距离
    distances_to_center = np.linalg.norm(points - center, axis=1)

    # 根据距离排序，找到应该保留的点的索引
    num_points_to_keep = int(len(points) * keep_ratio)
    sorted_indices = np.argsort(distances_to_center)
    indices_to_keep = sorted_indices[:num_points_to_keep]

    # 保留应该保留的点
    points_to_keep = points[indices_to_keep]

    return points_to_keep

def find_points_bbox(points, keep_ratio=0.95):
    """
    points: [N, 3]
    keep_ratio: 保留的点的比例
    """
    assert len(points.shape) == 2 and points.shape[1] == 3
    points = keep_points_near_center(points, keep_ratio=keep_ratio)
    scene_center = torch.from_numpy(points.mean(0)).cuda()
    scene_radius = 0.5 * np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))
    return scene_center.float(), scene_radius