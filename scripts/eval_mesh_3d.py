"""
3D metrics of an exported mesh against the Blender ground truth.

The network mesh comes out of ``--to_mesh`` in *canonical* space, while the ground truth
exported by ``scripts/blender_export_gt_mesh.py`` lives in the object local frame of a Blender
instance. ``utils/blender_align.py`` solves the transform between the two analytically -- the
shared camera fixes everything but the SfM scale, and the object poses fix that -- so no ICP
is needed and the numbers stay sensitive to a real pose error. ``--icp`` refines the analytic
alignment afterwards if you want the shape-only error.

Two frames are supported:

* ``blender_local`` (default): compare one canonical object against ``gt_mesh_local.ply``. Note
  that a Blender object may carry an anisotropic object scale, in which case distances in this
  frame are stretched along the object axes.
* ``blender_world``: compare against ``gt_mesh_world.ply``, i.e. the whole pile placed in the
  Blender world. This is the frame to quote metric numbers in.

Usage:
    python scripts/eval_mesh_3d.py \\
        --mesh exps/Mat-coffee-mesh/<timestamp>/mesh/mesh.ply \\
        --gt_mesh hf_data/train_split/coffee/gt/gt_mesh_local.ply \\
        --data_split_dir hf_data/train_split/coffee

    python scripts/eval_mesh_3d.py --mesh .../mesh.ply \\
        --gt_mesh .../gt_mesh_world.ply --frame blender_world \\
        --data_split_dir hf_data/train_split/coffee --icp
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils import blender_align, mesh_metrics


def parse_args() -> argparse.Namespace:
    """
    Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mesh', type=str, required=True,
                        help='mesh exported with --to_mesh (mesh.ply, mesh.obj or '
                             'mesh_attributes.npz), in canonical space')
    parser.add_argument('--gt_mesh', type=str, required=True,
                        help='ground truth mesh from scripts/blender_export_gt_mesh.py')
    parser.add_argument('--data_split_dir', type=str, default='',
                        help='processed object directory holding the Blender ground truth '
                             'poses; read from the export transforms.json if omitted')
    parser.add_argument('--frame', type=str, default='blender_local',
                        choices=['blender_local', 'blender_world', 'as_is'],
                        help='frame to compare in; "as_is" skips the alignment entirely, for '
                             'meshes that are already in the same frame')
    parser.add_argument('--instance', type=int, default=-1,
                        help='which instance to use for the alignment; -1 averages over all of '
                             'them (blender_local) or replicates the mesh at every instance '
                             '(blender_world)')
    parser.add_argument('--icp', default=False, action='store_true',
                        help='refine the analytic alignment with trimmed ICP; reports the pose '
                             'correction it applied, which is itself a diagnostic')
    parser.add_argument('--icp_scale', default=True, action=argparse.BooleanOptionalAction,
                        help='let the ICP refinement change the scale as well')
    parser.add_argument('--samples', type=int, default=200000,
                        help='surface samples drawn from each mesh')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.005, 0.01, 0.02],
                        help='F-score thresholds, as a fraction of the ground truth bounding '
                             'box diagonal')
    parser.add_argument('--seed', type=int, default=0, help='seed of the surface sampler')
    parser.add_argument('--output', type=str, default='',
                        help='where to write the json report (default: next to --mesh)')
    parser.add_argument('--save_aligned', type=str, default='',
                        help='optionally write the aligned prediction as a ply, to inspect it '
                             'next to the ground truth')
    return parser.parse_args()


def resolve_data_split_dir(mesh_path: str, given: str) -> str:
    """
    Find the dataset directory belonging to an exported mesh.

    Args:
        mesh_path: path of the exported mesh.
        given: value of ``--data_split_dir``, which wins if it is set.

    Returns:
        The dataset directory.

    Raises:
        SystemExit: if it can neither be given nor recovered from ``transforms.json``.
    """
    if given:
        return given
    transforms = os.path.join(os.path.dirname(os.path.abspath(mesh_path)), 'transforms.json')
    if os.path.exists(transforms):
        with open(transforms, 'r') as handle:
            recorded = json.load(handle).get('data_split_dir') or ''
        if recorded and os.path.isdir(recorded):
            print('took the dataset directory from {}'.format(transforms))
            return recorded
    raise SystemExit('pass --data_split_dir: it could not be recovered from the export')


def canonical_to_target(alignment: Dict[str, Any], frame: str,
                        instance: int) -> List[np.ndarray]:
    """
    Transforms that map the canonical mesh into the frame the ground truth lives in.

    Args:
        alignment: the dict returned by :func:`utils.blender_align.solve_from_data_dir`.
        frame: ``'blender_local'``, ``'blender_world'`` or ``'as_is'``.
        instance: index of the instance to use, or -1 for "all of them".

    Returns:
        One 4x4 matrix per copy of the mesh to place: a single matrix for
        ``blender_local``/``as_is``, and one per instance for ``blender_world`` with
        ``instance == -1``.

    Raises:
        ValueError: for an unknown frame.
    """
    if frame == 'as_is':
        return [np.eye(4, dtype=np.float64)]
    if frame == 'blender_local':
        if instance < 0:
            return [alignment['canonical_to_blender']]
        return [alignment['canonical_to_blender_per_instance'][instance]]
    if frame == 'blender_world':
        world = alignment['canonical_to_blender_world']
        return list(world) if instance < 0 else [world[instance]]
    raise ValueError('unknown frame {!r}'.format(frame))


def place_mesh(vertices: np.ndarray, faces: np.ndarray,
               transforms: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Place one copy of a mesh per transform and merge the copies into a single mesh.

    Args:
        vertices: ``[n,3]`` array.
        faces: ``[m,3]`` int array.
        transforms: 4x4 matrices to place the mesh with.

    Returns:
        The merged vertices ``[k*n,3]`` and faces ``[k*m,3]``.
    """
    all_vertices, all_faces = [], []
    for matrix in transforms:
        all_faces.append(faces + sum(len(v) for v in all_vertices))
        all_vertices.append(mesh_metrics.transform_points(vertices, matrix))
    return np.concatenate(all_vertices, axis=0), np.concatenate(all_faces, axis=0)


def describe_transform(matrix: np.ndarray) -> Dict[str, float]:
    """
    Summarise a 4x4 transform as a scale, a rotation angle and a translation length.

    Args:
        matrix: ``[4,4]`` transform.

    Returns:
        A dict with ``scale``, ``rotation_deg`` and ``translation``.
    """
    linear = np.asarray(matrix, dtype=np.float64)[:3, :3]
    singular = np.linalg.svd(linear, compute_uv=False)
    rotation = blender_align.polar_rotations(np.asarray(matrix, dtype=np.float64)[None])
    return {'scale': float(singular.mean()),
            'rotation_deg': float(blender_align.rotation_angle(rotation)[0]),
            'translation': float(np.linalg.norm(matrix[:3, 3]))}


def print_report(report: Dict[str, Any]) -> None:
    """
    Print the metric report in a fixed, readable order.

    Args:
        report: the dict produced by :func:`utils.mesh_metrics.evaluate_meshes`.
    """
    diagonal = report['diagonal']
    print('\nground truth bounding box diagonal: {:.5f}'.format(diagonal))
    print('{:<24s} {:>14s} {:>16s}'.format('metric', 'absolute', 'relative (%)'))
    for key in ['chamfer_l1', 'accuracy', 'completeness', 'accuracy_median',
                'completeness_median', 'hausdorff']:
        print('{:<24s} {:14.6f} {:15.3f}%'.format(key, report[key],
                                                  100.0 * report[key + '_relative']))
    print('{:<24s} {:14.8f} {:15.5f}%'.format('chamfer_l2', report['chamfer_l2'],
                                              100.0 * report['chamfer_l2_relative']))
    print('{:<24s} {:14.4f}'.format('normal_consistency', report['normal_consistency']))
    for key in sorted(k for k in report if k.startswith('f_score@')):
        threshold = key.split('@')[1]
        print('{:<24s} {:14.4f}   (precision {:.4f}, recall {:.4f})'.format(
            key, report[key], report['precision@' + threshold], report['recall@' + threshold]))


def main() -> None:
    """Align the exported mesh with the Blender ground truth and report the 3D metrics."""
    args = parse_args()

    vertices, faces = mesh_metrics.load_mesh(args.mesh)
    gt_vertices, gt_faces = mesh_metrics.load_mesh(args.gt_mesh)
    print('prediction   {}: {} vertices, {} faces'.format(args.mesh, len(vertices), len(faces)))
    print('ground truth {}: {} vertices, {} faces'.format(args.gt_mesh, len(gt_vertices),
                                                          len(gt_faces)))

    alignment: Dict[str, Any] = {}
    data_split_dir = ''
    if args.frame != 'as_is':
        data_split_dir = resolve_data_split_dir(args.mesh, args.data_split_dir)
        alignment = blender_align.solve_from_data_dir(data_split_dir, verbose=True)
    transforms = canonical_to_target(alignment, args.frame, args.instance)
    print('placing {} copy/copies of the prediction in the {} frame'.format(len(transforms),
                                                                           args.frame))
    vertices, faces = place_mesh(vertices, faces, transforms)

    refinement: Optional[Dict[str, float]] = None
    if args.icp:
        source, _ = mesh_metrics.sample_mesh_surface(vertices, faces, min(args.samples, 100000),
                                                     seed=args.seed)
        target, _ = mesh_metrics.sample_mesh_surface(gt_vertices, gt_faces,
                                                     min(args.samples, 100000), seed=args.seed + 1)
        matrix, rmse = mesh_metrics.icp_with_scale(source, target, with_scale=args.icp_scale)
        vertices = mesh_metrics.transform_points(vertices, matrix)
        refinement = describe_transform(matrix)
        refinement['rmse'] = rmse
        print('icp refinement: scale {scale:.5f}, rotation {rotation_deg:.3f} deg, '
              'translation {translation:.5f}, rmse {rmse:.6f}'.format(**refinement))

    report = mesh_metrics.evaluate_meshes((vertices, faces), (gt_vertices, gt_faces),
                                         samples=args.samples,
                                         f_score_thresholds=tuple(args.thresholds),
                                         seed=args.seed)
    print_report(report)

    report['mesh'] = os.path.abspath(args.mesh)
    report['gt_mesh'] = os.path.abspath(args.gt_mesh)
    report['frame'] = args.frame
    report['instance'] = int(args.instance)
    report['data_split_dir'] = data_split_dir
    report['icp'] = refinement
    if alignment:
        report['sfm_scale'] = float(alignment['sfm_scale'])
        report['pose_corner_spread'] = float(alignment['corner_spread'].max())
        report['pose_rotation_spread_deg'] = float(alignment['rotation_spread_deg'].max())

    if args.save_aligned:
        from utils import mesh_util
        mesh_util.write_ply(args.save_aligned, vertices, faces)
        print('\nwrote the aligned prediction to {}'.format(args.save_aligned))

    output = args.output or os.path.join(os.path.dirname(os.path.abspath(args.mesh)),
                                         'metrics_3d.json')
    with open(output, 'w') as handle:
        json.dump(report, handle, indent=2)
    print('wrote {}'.format(output))


if __name__ == '__main__':
    main()
