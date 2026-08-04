"""
Check that the SfM/network frames of a processed object line up with its Blender ground truth.

This is the offline half of "does a Blender render of the ground truth match the training
image": it verifies the pieces a render depends on -- the camera convention, the SfM to
Blender similarity solved in ``utils/blender_align.py``, the ground truth object poses and
the intrinsics -- without needing Blender itself. If the checks below pass, geometry
rendered from the ``.blend`` at the ground truth camera lands on the same pixels as the
training image; ``scripts/blender_render_gt.py`` then compares the shading as well.

Three checks are run:

1. *intrinsics*: project the COLMAP point cloud with (SfM camera, COLMAP focal) and with
   (Blender camera, Blender focal) after mapping the points through the solved similarity.
   The two only differ through the focal length, so this measures the COLMAP focal error in
   pixels.
2. *ground truth poses*: take each instance's point cloud, pull it into the canonical frame
   with the *predicted* pose, place it with the *ground truth* Blender pose and project it
   with the Blender camera. Points must land on their own instance in
   ``train/000_instance_seg.png``. This exercises the whole chain, including the averaged
   canonical-to-Blender transform used by the 3D metrics.
3. *exported mesh* (optional, ``--mesh``): the same test for the vertices of a mesh exported
   with ``--to_mesh``, plus an approximate silhouette IoU per instance.

Usage:
    python scripts/check_blender_alignment.py --data_split_dir hf_data/train_split/coffee
    python scripts/check_blender_alignment.py --data_split_dir hf_data/train_split/coffee \\
        --mesh exps/Mat-coffee-mesh/<timestamp>/mesh/mesh.ply
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio.v2 as imageio
import numpy as np

from utils import blender_align, rend_util


def parse_args() -> argparse.Namespace:
    """
    Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_split_dir', type=str, required=True,
                        help='processed object directory, e.g. hf_data/train_split/coffee')
    parser.add_argument('--mesh', type=str, default='',
                        help='optional mesh.ply / mesh_attributes.npz exported with --to_mesh')
    parser.add_argument('--output', type=str, default='',
                        help='where to write the overlay figure (default: alignment_check.png '
                             'inside --data_split_dir)')
    parser.add_argument('--resolution', type=int, default=0,
                        help='image side length; read from the segmentation image by default')
    return parser.parse_args()


def load_point_clouds(data_split_dir: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Read the COLMAP point cloud of an object, in the SfM world frame.

    Args:
        data_split_dir: processed object directory.

    Returns:
        all_points: ``[n,3]`` every point of the pile.
        per_instance: one ``[n_i,3]`` array per registered instance, in the order of
            ``non_empty_indexes.txt``.
    """
    points = np.load(os.path.join(data_split_dir, 'points_world.npy'), allow_pickle=True).item()
    per_instance, index = [], 0
    while 'points_world_{}'.format(index) in points:
        per_instance.append(np.asarray(points['points_world_{}'.format(index)], dtype=np.float64))
        index += 1
    return np.asarray(points['points_world_all'], dtype=np.float64), per_instance


def load_mesh_vertices(path: str) -> np.ndarray:
    """
    Read the vertices of an exported mesh, in canonical space.

    Args:
        path: a ``mesh.ply`` or a ``mesh_attributes.npz`` written by ``--to_mesh``.

    Returns:
        ``[n,3]`` float64 array of vertices.

    Raises:
        ValueError: for an unsupported file extension.
    """
    if path.endswith('.npz'):
        return np.asarray(np.load(path)['vertices'], dtype=np.float64)
    if path.endswith('.ply'):
        import trimesh
        return np.asarray(trimesh.load(path, process=False).vertices, dtype=np.float64)
    raise ValueError('expected a .ply or .npz mesh, got {}'.format(path))


def label_statistics(pixels: np.ndarray, depth: np.ndarray, segmentation: np.ndarray,
                     expected_label: int) -> Dict[str, float]:
    """
    Where do projected points land in the instance segmentation?

    Args:
        pixels: ``[n,2]`` projected ``(column, row)`` coordinates.
        depth: ``[n]`` depth along the view direction; points behind the camera are dropped.
        segmentation: ``[h,w]`` int label image, 0 = background, ``i+1`` = instance ``i``.
        expected_label: the label the points should land on.

    Returns:
        A dict with the fraction of points on the expected instance, on any instance and
        outside the image.
    """
    height, width = segmentation.shape
    column = np.rint(pixels[:, 0]).astype(int)
    row = np.rint(pixels[:, 1]).astype(int)
    inside = (depth > 0) & (column >= 0) & (column < width) & (row >= 0) & (row < height)
    if not inside.any():
        return {'on_instance': 0.0, 'on_any_object': 0.0, 'outside_image': 1.0}
    labels = segmentation[row[inside], column[inside]]
    return {'on_instance': float((labels == expected_label).mean()),
            'on_any_object': float((labels > 0).mean()),
            'outside_image': float(1.0 - inside.mean())}


def splat_silhouette(pixels: np.ndarray, depth: np.ndarray, shape: Tuple[int, int],
                     radius: int = 1) -> np.ndarray:
    """
    Approximate the silhouette of a projected point set by splatting and filling holes.

    A real silhouette needs a rasteriser; this is only meant as a coarse IoU indicator.

    Args:
        pixels: ``[n,2]`` projected ``(column, row)`` coordinates.
        depth: ``[n]`` depth along the view direction.
        shape: ``(height, width)`` of the image.
        radius: half width of the square splat, in pixels.

    Returns:
        ``[h,w]`` boolean silhouette.
    """
    from scipy import ndimage

    height, width = shape
    canvas = np.zeros(shape, dtype=bool)
    column = np.rint(pixels[:, 0]).astype(int)
    row = np.rint(pixels[:, 1]).astype(int)
    keep = (depth > 0) & (column >= 0) & (column < width) & (row >= 0) & (row < height)
    canvas[row[keep], column[keep]] = True
    if radius > 0:
        canvas = ndimage.binary_dilation(canvas, np.ones([2 * radius + 1] * 2, dtype=bool))
    return ndimage.binary_fill_holes(canvas)


def intersection_over_union(prediction: np.ndarray, target: np.ndarray) -> float:
    """
    IoU of two boolean masks.

    Args:
        prediction: ``[h,w]`` boolean mask.
        target: ``[h,w]`` boolean mask.

    Returns:
        The intersection over union, 0 if both masks are empty.
    """
    union = np.logical_or(prediction, target).sum()
    return 0.0 if union == 0 else float(np.logical_and(prediction, target).sum() / union)


def save_overlay(path: str, image: np.ndarray, projections: List[np.ndarray],
                 segmentation: np.ndarray) -> str:
    """
    Write a figure with the training image, the projected points and the segmentation.

    Args:
        path: destination png.
        image: ``[h,w,3]`` display ready training image in ``[0,1]``.
        projections: one ``[n_i,2]`` array of pixel coordinates per instance.
        segmentation: ``[h,w]`` label image.

    Returns:
        The path that was written.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title('training image')
    axes[1].imshow(image)
    colours = plt.get_cmap('tab20')(np.linspace(0, 1, max(len(projections), 2)))
    for index, pixels in enumerate(projections):
        axes[1].scatter(pixels[:, 0], pixels[:, 1], s=1.5, color=colours[index],
                        linewidths=0)
    axes[1].set_title('gt pose + blender camera')
    axes[2].imshow(segmentation, cmap='tab20')
    axes[2].set_title('instance segmentation')
    for axis in axes:
        axis.set_xlim(0, image.shape[1])
        axis.set_ylim(image.shape[0], 0)
        axis.axis('off')
    figure.tight_layout()
    figure.savefig(path, dpi=120, bbox_inches='tight', pad_inches=0.05)
    plt.close(figure)
    return path


def main() -> None:
    """Run every check and print a report."""
    args = parse_args()
    data_split_dir = args.data_split_dir

    alignment = blender_align.solve_from_data_dir(data_split_dir, verbose=True)
    indexes = alignment['non_empty_indexes']
    scale_matrix = alignment['scale_matrix']
    sfm_object_poses = blender_align.load_object_poses(
        os.path.join(data_split_dir, 'object_pred_pose.json'))
    blender_object_poses = blender_align.load_object_poses(
        os.path.join(data_split_dir, 'blender_object_gt_pose.json'), indexes=indexes)
    canonical_to_blender = alignment['canonical_to_blender']

    segmentation_path = os.path.join(data_split_dir, 'train', '000_instance_seg.png')
    segmentation = rend_util.load_seg(segmentation_path, input_range='0_255',
                                      output_range='0_n', non_empty_indexes=indexes,
                                      same_obj_num=int(indexes.max()) + 1).astype(int)
    resolution = args.resolution or segmentation.shape[0]

    all_points, per_instance = load_point_clouds(data_split_dir)

    print('\n[1] intrinsics: sfm camera + colmap focal vs blender camera + blender focal')
    sfm_pixels, sfm_depth = blender_align.project_points_to_pixels(
        all_points, alignment['sfm_camera'], alignment['focal'], resolution)
    blender_points = all_points @ alignment['world_to_blender'][:3, :3].T \
        + alignment['world_to_blender'][:3, 3][None]
    blender_pixels, _ = blender_align.project_points_to_pixels(
        blender_points, alignment['blender_camera'], alignment['blender_focal'], resolution)
    difference = np.linalg.norm(sfm_pixels - blender_pixels, axis=1)
    print('    reprojection difference: mean {:.2f} px, median {:.2f} px, max {:.2f} px'.format(
        float(difference.mean()), float(np.median(difference)), float(difference.max())))
    print('    colmap focal {:.1f} vs blender focal {:.1f} ({:+.2f}%)'.format(
        alignment['focal'], alignment['blender_focal'],
        100.0 * (alignment['focal'] / alignment['blender_focal'] - 1.0)))
    statistics = label_statistics(sfm_pixels, sfm_depth, segmentation, expected_label=-1)
    print('    colmap points inside some instance: {:.1%} (sanity check of the seg mask)'
          .format(statistics['on_any_object']))

    print('\n[2] ground truth poses: canonical point cloud placed by the blender gt pose')
    projections: List[np.ndarray] = []
    scores: List[float] = []
    canonical_to_world = np.linalg.inv(sfm_object_poses @ scale_matrix[None])
    for index in range(len(per_instance)):
        canonical = per_instance[index] @ canonical_to_world[index][:3, :3].T \
            + canonical_to_world[index][:3, 3][None]
        placed = canonical @ canonical_to_blender[:3, :3].T + canonical_to_blender[:3, 3][None]
        placed = placed @ blender_object_poses[index][:3, :3].T \
            + blender_object_poses[index][:3, 3][None]
        pixels, depth = blender_align.project_points_to_pixels(
            placed, alignment['blender_camera'], alignment['blender_focal'], resolution)
        statistics = label_statistics(pixels, depth, segmentation, expected_label=index + 1)
        scores.append(statistics['on_instance'])
        projections.append(pixels)
        print('    instance {:2d} (blender id {:2d}): on its own instance {:6.1%}, '
              'on any instance {:6.1%}'.format(index, indexes[index],
                                               statistics['on_instance'],
                                               statistics['on_any_object']))
    print('    mean over instances: {:.1%} of the points land on their own instance'
          .format(float(np.mean(scores))))

    if args.mesh:
        print('\n[3] exported mesh: {}'.format(args.mesh))
        vertices = load_mesh_vertices(args.mesh)
        mesh_scores, ious = [], []
        for index in range(len(per_instance)):
            placed = vertices @ canonical_to_blender[:3, :3].T + canonical_to_blender[:3, 3][None]
            placed = placed @ blender_object_poses[index][:3, :3].T \
                + blender_object_poses[index][:3, 3][None]
            pixels, depth = blender_align.project_points_to_pixels(
                placed, alignment['blender_camera'], alignment['blender_focal'], resolution)
            statistics = label_statistics(pixels, depth, segmentation, expected_label=index + 1)
            silhouette = splat_silhouette(pixels, depth, segmentation.shape)
            iou = intersection_over_union(silhouette, segmentation == index + 1)
            mesh_scores.append(statistics['on_instance'])
            ious.append(iou)
            print('    instance {:2d}: vertices on its own instance {:6.1%}, '
                  'approximate silhouette IoU {:.3f}'.format(index, statistics['on_instance'],
                                                             iou))
        print('    mean: on own instance {:.1%}, IoU {:.3f}'.format(
            float(np.mean(mesh_scores)), float(np.mean(ious))))

    image_path = os.path.join(data_split_dir, 'train', '000_rgb.exr')
    if os.path.exists(image_path):
        image = np.clip(rend_util.load_exr(image_path), 0.0, None) ** (1.0 / 2.2)
    else:
        image = imageio.imread(os.path.join(data_split_dir, 'train', '000_rgb.png'))[..., :3] / 255.0
    output = args.output or os.path.join(data_split_dir, 'alignment_check.png')
    print('\nwrote {}'.format(save_overlay(output, np.clip(image, 0, 1), projections,
                                          segmentation)))


if __name__ == '__main__':
    main()
