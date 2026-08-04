"""
Rebuild a broken ``blender_object_gt_pose.json`` from the geometry of its ``.blend``.

Must be run by Blender:

    LD_LIBRARY_PATH=/opt/py313/lib /opt/py313/bin/python scripts/blender_fix_gt_poses.py -- \\
        --blend_file blender_data/clock/clock.blend \\
        --data_split_dir hf_data/train_split/clock

Eight of the nine released objects record poses that *are* the poses of their scene: matching the
pile against ``blender_object_gt_pose.json`` leaves a residual of ~1e-7 Blender units
(:func:`scripts.blender_common.match_blocks_to_instances`). ``clock`` does not -- its residual is
1.86, no permutation of its recorded matrices fits its own geometry, and their pairwise distance
spectrum differs from the scene's, so the file describes a *different* arrangement of the same nine
clocks and cannot be repaired by relabelling or by a global transform. Its per-instance ground
truth is therefore unusable: ``check_blender_alignment.py`` puts only 38 % of the COLMAP points on
the instance they were assigned to and reports a 103 deg SfM rotation spread.

The scene itself is right -- it renders to the training image at a 0.9993 silhouette IoU and a
0.00 px shift -- so the poses are recovered from it instead:

1. cut the joined pile into contiguous vertex blocks, one per instance
   (:func:`scripts.blender_common.contiguous_blocks`); the blocks share their vertex order, so
   :func:`scripts.blender_common.rigid_transform` gives the *exact* transform between any two.
2. decide which block is which instance by splatting each block into
   ``train/000_instance_seg.png`` with the ground truth camera and taking the segmentation label it
   covers. This is the only step that needs an outside source of truth, and it is the same
   convention ``utils/rend_util.load_seg`` uses: the grey levels are ``255*k/n``, so the k-th level
   in ascending order is object index ``k-1``. ``--force`` turns the step into a self-test on the
   four joined scenes whose pose files *are* sound, and it reproduces the labelling those files
   imply on every block of all four (``coffee`` 7/7, ``fire`` 10/10, ``gitar`` 9/9, ``tin`` 9/9),
   which is what licenses using it on ``clock``.
3. fix the remaining gauge freedom. Only the products ``M_i @ inv(M_j)`` are observable -- writing
   ``M_i @ C`` for one fixed rigid ``C`` moves the local frame but leaves every quantity that uses
   the poses unchanged, because ``utils/blender_align.py`` only ever evaluates ``M_i @ T`` with
   ``T = inv(M_i) @ ...`` -- so ``C`` is chosen to put the origin of the local frame at the centre
   of the instance bounding box, at unit scale. The released files use unit scale too on the
   objects whose Blender scale is 1.

``--min_purity`` guards step 2: it refuses to write anything when a block does not mostly cover the
label it was given. It is a guard, not a measure of how sure the assignment is -- the runner-up
label is always at least an order of magnitude behind -- and ``tin`` trips it at 51.6 % because its
cans are 352 vertices apiece and largely hidden, even though its labelling still comes out right.

The result is written with a ``.orig`` backup. Validate it without Blender afterwards:

    python scripts/check_blender_alignment.py --data_split_dir hf_data/train_split/clock
    ... --python scripts/blender_export_gt_mesh.py -- ... --all_instances
"""

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blender_common import (JOINED, balanced_assignment, contiguous_blocks, load_gt_poses,
                            match_blocks_to_instances, resolve_instances, rigid_transform)
from blender_export_gt_mesh import evaluate_mesh


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse the arguments that follow ``--`` on Blender's command line.

    Args:
        argv: argument list to parse; taken from ``sys.argv`` after ``--`` by default.

    Returns:
        The parsed arguments.
    """
    if argv is None:
        argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--blend_file', type=str, default='',
                        help='.blend to open; only needed when this runs through the bpy pip '
                             'module, since "blender --background <file>.blend" already opened it')
    parser.add_argument('--data_split_dir', type=str, required=True,
                        help='processed object directory; its blender_object_gt_pose.json is the '
                             'file that gets rebuilt and its blender_camera_gt_pose.json / '
                             'instance segmentation label the blocks')
    parser.add_argument('--output', type=str, default='',
                        help='where to write the rebuilt poses; empty overwrites '
                             'blender_object_gt_pose.json in the split directory')
    parser.add_argument('--segmentation', type=str, default='train/000_instance_seg.png',
                        help='instance segmentation image, relative to --data_split_dir, whose '
                             'labels name the instances')
    parser.add_argument('--frame', type=int, default=0,
                        help='index of the frame in blender_camera_gt_pose.json that the '
                             'segmentation image belongs to')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                        help='largest match residual, in blender units, still counted as "the '
                             'recorded poses describe this scene"; a file within it is left alone '
                             'unless --force says otherwise')
    parser.add_argument('--min_purity', type=float, default=0.6,
                        help='smallest fraction of a block\'s labelled pixels that must carry the '
                             'label it is assigned to; below it the labelling is refused')
    parser.add_argument('--splat_points', type=int, default=50000,
                        help='number of surface points per block used to splat it into the '
                             'segmentation image; the vertices alone are used if there are '
                             'already that many')
    parser.add_argument('--force', default=False, action='store_true',
                        help='rebuild even when the recorded poses already fit the scene')
    parser.add_argument('--dry_run', default=False, action='store_true',
                        help='report what would be written without touching any file')
    return parser.parse_args(argv)


def load_segmentation(path: str) -> Any:
    """
    Read an 8 bit instance segmentation image through Blender.

    Blender's bundled Python has numpy but no imageio, so the image is loaded as a Blender
    datablock. Its colour space is forced to ``Non-Color`` to stop the sRGB transform, which turns
    the float pixels back into the stored bytes, and the rows are flipped because Blender stores
    images bottom-up.

    Args:
        path: image file to read.

    Returns:
        ``[h,w]`` uint8 array of grey levels.
    """
    import bpy
    import numpy as np

    image = bpy.data.images.load(os.path.abspath(path))
    image.colorspace_settings.name = 'Non-Color'
    width, height = image.size
    buffer = np.empty(width * height * image.channels, dtype=np.float32)
    image.pixels.foreach_get(buffer)
    grey = buffer.reshape(height, width, image.channels)[::-1, :, 0]
    bpy.data.images.remove(image)
    return np.rint(grey * 255.0).astype(np.uint8)


def instance_labels(segmentation: Any, count: int) -> Any:
    """
    Turn grey levels into instance indexes, the way ``utils/rend_util.load_seg`` does.

    The released images store instance ``k`` (1-based, 0 is the background) as grey level
    ``round(255*k/n)``, so the k-th distinct non-zero level in ascending order is object index
    ``k-1``. The compaction ``load_seg`` applies for the unregistered instances happens at load
    time and does not touch the file, so every instance of the scene is present here.

    Args:
        segmentation: ``[h,w]`` array of grey levels.
        count: number of instances the scene holds.

    Returns:
        ``[h,w]`` int array holding the object index of every pixel, -1 for the background.

    Raises:
        ValueError: if the image does not hold exactly ``count`` distinct non-zero levels, in which
            case the label convention cannot be trusted.
    """
    import numpy as np

    levels = np.unique(segmentation)
    levels = levels[levels > 0]
    if len(levels) != count:
        raise ValueError('the segmentation image holds {} instance labels but the pose file has '
                         '{} instances'.format(len(levels), count))
    index_of_level = np.full(256, -1, dtype=np.int64)
    index_of_level[levels] = np.arange(count)
    return index_of_level[segmentation]


def barycentric_lattice(order: int) -> Any:
    """
    Barycentric weights of a regular triangular lattice, one row per sample.

    Args:
        order: number of samples along a triangle edge; ``order`` 1 is the centroid only.

    Returns:
        ``[order*(order+1)/2,3]`` array of weights summing to 1.
    """
    import numpy as np

    if order <= 1:
        return np.full((1, 3), 1.0 / 3.0)
    steps = (np.arange(order) + 0.5) / order
    first, second = np.meshgrid(steps, steps, indexing='ij')
    keep = first + second <= 1.0
    first, second = first[keep], second[keep]
    return np.stack([1.0 - first - second, first, second], axis=-1)


def surface_points(vertices: Any, faces: Any, minimum: int) -> Any:
    """
    Spread points over the triangles of a mesh, deterministically.

    The block labelling below splats points into the segmentation image, and a sparse mesh does not
    cover enough pixels for that to be reliable -- ``tin`` has only 352 vertices per instance, which
    hit ~340 of the ~3000 pixels the instance occupies, so a handful of stray hits already dominate.
    Sampling the faces instead makes the coverage a property of the surface rather than of the
    tessellation. The lattice is regular, not random, so the result does not depend on a seed.

    Args:
        vertices: ``[n,3]`` array of vertices.
        faces: ``[m,3]`` int array of triangles.
        minimum: number of points to aim for; the vertices alone are used once they reach it.

    Returns:
        ``[k,3]`` array of points on the surface, the vertices included.
    """
    import numpy as np

    if len(vertices) >= minimum or len(faces) == 0:
        return np.asarray(vertices, dtype=np.float64)
    order = 1
    while order * (order + 1) // 2 * len(faces) < minimum:
        order += 1
    weights = barycentric_lattice(order)
    corners = np.asarray(vertices, dtype=np.float64)[np.asarray(faces)]
    samples = np.einsum('sw,fwc->fsc', weights, corners).reshape(-1, 3)
    return np.concatenate([np.asarray(vertices, dtype=np.float64), samples], axis=0)


def splat_blocks(blocks: List[Any], camera_to_world: Any, focal: float, resolution: int) -> Any:
    """
    Rasterise one block per pixel by z-buffering a point set.

    A depth buffer over projected points is enough to say which block owns a pixel, as long as the
    points cover the surface densely enough that the visible part of every block is hit many times
    over -- that is what :func:`surface_points` is for.

    Args:
        blocks: one ``[n,3]`` array of world space points per block.
        camera_to_world: ``[4,4]`` camera-to-world matrix in the Blender world frame.
        focal: focal length in pixels.
        resolution: side length of the square image.

    Returns:
        ``[resolution,resolution]`` int array holding the index of the nearest block that covers
        each pixel, -1 where nothing does.
    """
    import numpy as np

    from utils.blender_align import project_points_to_pixels

    depth_buffer = np.full((resolution, resolution), np.inf)
    owner = np.full((resolution, resolution), -1, dtype=np.int64)
    for block, points in enumerate(blocks):
        pixels, depth = project_points_to_pixels(points, camera_to_world, focal, resolution)
        column = np.rint(pixels[:, 0]).astype(np.int64)
        row = np.rint(pixels[:, 1]).astype(np.int64)
        inside = ((depth > 0) & (column >= 0) & (column < resolution) &
                  (row >= 0) & (row < resolution))
        order = np.argsort(-depth[inside])  # farthest first, so the nearest point writes last
        row, column, depth = row[inside][order], column[inside][order], depth[inside][order]
        closer = depth < depth_buffer[row, column]
        depth_buffer[row[closer], column[closer]] = depth[closer]
        owner[row[closer], column[closer]] = block
    return owner


def label_blocks(blocks: List[Any], labels: Any, camera_to_world: Any, focal: float) -> Tuple[
        Any, Any, Any]:
    """
    Decide which instance index each block of the joined pile carries.

    Args:
        blocks: one ``[n,3]`` array of world space surface points per block.
        labels: ``[h,w]`` int array of object indexes, -1 for the background.
        camera_to_world: ``[4,4]`` camera-to-world matrix in the Blender world frame.
        focal: focal length in pixels.

    Returns:
        assignment: ``[k]`` int array giving the instance index of each block; a permutation,
            because every instance takes exactly one block.
        purity: ``[k]`` float array, the fraction of each block's labelled pixels that carry the
            label it was given. Below 1 because the blocks occlude each other and a splat is not a
            rasteriser, not because the assignment is in doubt -- the runner-up is an order of
            magnitude behind.
        confusion: ``[k,k]`` pixel counts, blocks by instance indexes, for the report.
    """
    import numpy as np

    owner = splat_blocks(blocks, camera_to_world, focal, labels.shape[0])
    both = (owner >= 0) & (labels >= 0)
    confusion = np.zeros((len(blocks), len(blocks)), dtype=np.int64)
    np.add.at(confusion, (owner[both], labels[both]), 1)
    assignment = balanced_assignment(-confusion.astype(np.float64), 1)
    covered = np.maximum(confusion.sum(axis=1), 1)
    purity = confusion[np.arange(len(blocks)), assignment] / covered
    return assignment, purity, confusion


def poses_from_blocks(blocks: List[Any], assignment: Any) -> Any:
    """
    Build one ``matrix_world`` per instance from the exact transforms between the blocks.

    The blocks are duplicates joined in one go, so they share their vertex order and
    :func:`scripts.blender_common.rigid_transform` recovers the transform between any two of them
    exactly. That fixes every pose up to one common right factor ``C`` (see the module docstring),
    which is spent on making the local frame the geometry of instance 00 centred on its own
    bounding box: ``M_00`` comes out as a pure translation and every other ``M_i`` as the rigid
    motion that carries instance 00 onto instance ``i``, at unit scale.

    Args:
        blocks: one ``[n,3]`` array of world space vertices per block, in matched vertex order.
        assignment: ``[k]`` int array giving the instance index of each block.

    Returns:
        ``[k,4,4]`` float64 array of object-to-world matrices, in instance order.
    """
    import numpy as np

    relative = np.array([rigid_transform(blocks[0], block) for block in blocks])
    anchor = int(np.nonzero(assignment == 0)[0][0])
    centre = 0.5 * (blocks[anchor].min(axis=0) + blocks[anchor].max(axis=0))
    gauge = np.eye(4, dtype=np.float64)
    gauge[:3, 3] = centre
    local_to_block_frame = np.linalg.inv(relative[anchor]) @ gauge

    matrices = np.zeros((len(blocks), 4, 4), dtype=np.float64)
    for block, instance in enumerate(assignment):
        matrices[instance] = relative[block] @ local_to_block_frame
    return matrices


def report_matrices(names: List[str], matrices: Any) -> Dict[str, Any]:
    """
    Sanity check a freshly built pose set: unit scale, right handed, no duplicates.

    Args:
        names: instance names, in instance order.
        matrices: ``[k,4,4]`` object-to-world matrices, in instance order.

    Returns:
        ``scale_deviation`` (largest ``|singular value - 1|``), ``determinant`` (smallest
        rotation determinant) and ``closest_pair`` (smallest distance between two origins).
    """
    import numpy as np

    singular = np.linalg.svd(matrices[:, :3, :3], compute_uv=False)
    origins = matrices[:, :3, 3]
    distance = np.linalg.norm(origins[:, None] - origins[None], axis=-1)
    distance[np.arange(len(names)), np.arange(len(names))] = np.inf
    return {'scale_deviation': float(np.abs(singular - 1.0).max()),
            'determinant': float(min(np.linalg.det(matrix[:3, :3]) for matrix in matrices)),
            'closest_pair': float(distance.min())}


def main() -> None:
    """Rebuild the ground truth poses of a joined pile from its scene geometry."""
    try:
        import bpy
        import numpy as np
    except ImportError:
        raise SystemExit('this needs Blender: either\n'
                         '  blender --background <object>.blend --python {} -- --help\n'
                         'or the pip module, which brings its own Blender:\n'
                         '  pip install bpy && python {} -- --blend_file <object>.blend --help'
                         .format(os.path.relpath(__file__), os.path.relpath(__file__)))
    from utils.blender_align import focal_from_camera_angle, load_json

    args = parse_args()
    if args.blend_file:
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend_file))

    instances, layout = resolve_instances(args.data_split_dir, include_unregistered=True)
    if layout != JOINED:
        raise SystemExit('this scene holds one object per instance, so its poses are the object '
                         'matrices themselves and there is nothing to rebuild; '
                         'blender_render_gt.py --reapply_gt_poses checks those')

    poses = load_gt_poses(args.data_split_dir)
    names = sorted(poses)
    recorded = np.array([np.asarray(poses[name], dtype=np.float64) for name in names])

    joined = instances[0]['object']
    matrix_world = [list(row) for row in bpy.data.objects[joined].matrix_world]
    vertices, faces = evaluate_mesh(joined, matrix=matrix_world)
    block_of = contiguous_blocks(len(vertices), faces, len(names))
    blocks = [vertices[block_of == block] for block in range(len(names))]
    print('cut {} ({} vertices) into {} blocks of {}'.format(joined, len(vertices), len(names),
                                                             len(blocks[0])))
    splats = []
    for block in range(len(names)):
        keep = block_of == block
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[keep] = np.arange(int(keep.sum()))
        splats.append(surface_points(blocks[block], remap[faces[keep[faces].all(axis=1)]],
                                     args.splat_points))

    recorded_assignment, residual = match_blocks_to_instances(blocks, recorded)
    print('the recorded poses fit this scene to {:.3e} blender units'.format(residual))
    if residual <= args.tolerance and not args.force:
        raise SystemExit('that is within --tolerance {:.0e}, so the recorded poses already are the '
                         'poses of this scene and rebuilding them would only change the local '
                         'frame; pass --force to do it anyway'.format(args.tolerance))

    camera = load_json(os.path.join(args.data_split_dir, 'blender_camera_gt_pose.json'))
    camera_to_world = np.asarray(camera['frames'][args.frame]['transform_matrix'],
                                 dtype=np.float64)
    segmentation = load_segmentation(os.path.join(args.data_split_dir, args.segmentation))
    labels = instance_labels(segmentation, len(names))
    focal = focal_from_camera_angle(float(camera['camera_angle_x']), segmentation.shape[1])
    print('labelling the blocks in {} at focal {:.2f} px'.format(args.segmentation, focal))

    assignment, purity, confusion = label_blocks(splats, labels, camera_to_world, focal)
    for block, instance in enumerate(assignment):
        runner_up = np.sort(confusion[block])[-2]
        print('  block {} -> {} ({:.1%} of {} labelled pixels, runner-up {})'
              .format(block, names[instance], purity[block], confusion[block].sum(), runner_up))
    if residual <= args.tolerance:
        agree = int((assignment == recorded_assignment).sum())
        print('self-test: the recorded poses of this object do fit the scene, and the '
              'segmentation labels {} of the {} blocks the same way they do{}'
              .format(agree, len(names), '' if agree == len(names) else
                      ' -- recorded {}, segmentation {}'.format(recorded_assignment.tolist(),
                                                                assignment.tolist())))
    if purity.min() < args.min_purity:
        raise SystemExit('block {} agrees with its label on only {:.1%} of its pixels, below '
                         '--min_purity {:.0%}; the labelling is not trustworthy and no file was '
                         'written'.format(int(purity.argmin()), purity.min(), args.min_purity))

    matrices = poses_from_blocks(blocks, assignment)
    report = report_matrices(names, matrices)
    print('rebuilt poses: scale deviation {scale_deviation:.2e}, smallest determinant '
          '{determinant:.6f}, closest pair of origins {closest_pair:.4f}'.format(**report))
    _, new_residual = match_blocks_to_instances(blocks, matrices)
    print('the rebuilt poses fit this scene to {:.3e} blender units (was {:.3e})'
          .format(new_residual, residual))
    if new_residual > args.tolerance:
        raise SystemExit('the rebuilt poses do not fit the geometry they were built from, which '
                         'should be impossible; no file was written')

    path = args.output or os.path.join(args.data_split_dir, 'blender_object_gt_pose.json')
    if args.dry_run:
        print('--dry_run, so {} was left alone'.format(path))
        return
    if not args.output and not os.path.exists(path + '.orig'):
        shutil.copyfile(path, path + '.orig')
        print('kept the original as {}.orig'.format(path))
    with open(path, 'w') as handle:
        json.dump({name: matrices[index].tolist() for index, name in enumerate(names)}, handle,
                  indent=4)
    print('wrote {}'.format(path))
    print('check it with:\n  python scripts/check_blender_alignment.py --data_split_dir {}'
          .format(args.data_split_dir))


if __name__ == '__main__':
    main()
