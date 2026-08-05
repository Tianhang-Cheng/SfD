#!/usr/bin/env python3
"""
Turn the exported ``mesh.ply`` of every object into a small ``.glb`` the report page can display.

The meshes ``exp_runner.py --to_mesh`` writes are marching-cubes output at resolution 512: ~0.6 M
vertices / 1.3 M faces, 17-49 MB of binary ply each. That is too heavy to serve from a static page,
so each mesh is

1. decimated with a quadric edge collapse to ``--target_faces``,
2. re-coloured *and* re-normalled by nearest neighbour from the original vertices (the decimator
   moves vertices, so per-vertex attributes cannot be carried over by index; taking the normals
   from the dense mesh rather than recomputing them on the decimated one keeps the shading smooth
   instead of faceted),
3. rotated upright where the ground truth says which way is up (see below), scaled into a unit box
   and centred, so the page needs no per-object camera,
4. written as ``<output_dir>/<object>/mesh.glb`` with the vertex colours baked in.

Which way is up: nothing in the reconstruction knows. The canonical frame of the network comes from
the SfM reconstruction, whose axes are arbitrary. For the nine synthetic objects the Blender scene
does know, and ``utils/blender_align`` solves the canonical -> Blender-object transform, so those
are rotated gravity-up (Blender is z-up, gl is y-up). The six real-world captures have no Blender
scene and stay in the canonical frame -- they come out tilted, which is why the viewer lets you
drag. ``upright: true/false`` in the written index says which is which.

Usage (from the results checkout, after cmd_eval.sh has exported the meshes):

    python build_meshes.py --exps_dir /mnt/task_runtime/SfD/exps
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

SAMPLES = ['airplane', 'box', 'cake', 'cash', 'cheese', 'cleaner', 'clock',
           'coffee', 'cola', 'fire', 'gitar', 'potato', 'sign', 'tin', 'yogurt']


def parse_args() -> argparse.Namespace:
    """
    Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sfd_dir', type=str, default='/mnt/task_runtime/SfD',
                        help='the SfD checkout, for utils/blender_align and the dataset')
    parser.add_argument('--exps_dir', type=str, default='/mnt/task_runtime/SfD/exps',
                        help='where the Mat-<object>-mesh runs are')
    parser.add_argument('--data_root', type=str, default='/mnt/task_runtime/SfD/hf_data/train_split',
                        help='dataset root, for the poses that define "up"')
    parser.add_argument('--output_dir', type=str, default=os.path.join(HERE, 'assets'),
                        help='written as <output_dir>/<object>/mesh.glb')
    parser.add_argument('--index', type=str, default=os.path.join(HERE, 'meshes.json'),
                        help='json index of what was written')
    parser.add_argument('--target_faces', type=int, default=30000,
                        help='faces to decimate to; ~26 bytes of glb per face')
    parser.add_argument('--objects', type=str, nargs='*', default=SAMPLES,
                        help='objects to convert')
    return parser.parse_args()


def latest_mesh(exps_dir: str, name: str) -> Optional[str]:
    """
    Find the newest exported mesh of an object.

    Args:
        exps_dir: the ``exps/`` directory.
        name: object name.

    Returns:
        Path of ``mesh.ply``, or None if the object has no mesh export.
    """
    run_dir = os.path.join(exps_dir, 'Mat-{}-mesh'.format(name))
    if not os.path.isdir(run_dir):
        return None
    for stamp in sorted(os.listdir(run_dir), reverse=True):
        candidate = os.path.join(run_dir, stamp, 'mesh', 'mesh.ply')
        if os.path.isfile(candidate):
            return candidate
    return None


def upright_rotation(sfd_dir: str, data_root: str, name: str) -> Optional[np.ndarray]:
    """
    Rotation that takes the canonical frame to a gravity-up, gl-style (y-up) frame.

    ``blender_align`` gives the canonical -> Blender-object-local similarity; its rotation part maps
    canonical to Blender axes (z up), and the extra x(-90 deg) maps Blender z-up to gl y-up.

    Args:
        sfd_dir: the SfD checkout (added to ``sys.path`` for ``utils.blender_align``).
        data_root: dataset root holding ``<name>/``.
        name: object name.

    Returns:
        A 3x3 rotation, or None when the object has no Blender ground truth to define up.
    """
    if sfd_dir not in sys.path:
        sys.path.insert(0, sfd_dir)
    try:
        from utils import blender_align
    except ImportError:
        return None
    data_dir = os.path.join(data_root, name)
    if not os.path.isfile(os.path.join(data_dir, 'blender_camera_gt_pose.json')):
        return None
    try:
        alignment = blender_align.solve_from_data_dir(data_dir, verbose=False)
        canonical_to_blender = np.asarray(alignment['canonical_to_blender'], dtype=np.float64)
        rotation = blender_align.polar_rotations(canonical_to_blender[None])[0][:3, :3]
    except Exception as error:                       # a missing/broken pose file is not fatal here
        print('  {}: no upright rotation ({})'.format(name, error))
        return None
    blender_to_gl = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    return blender_to_gl @ rotation


def decimate(vertices: np.ndarray, faces: np.ndarray,
             target_faces: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapse edges until the mesh has about ``target_faces`` faces.

    Args:
        vertices: (V, 3) float vertex positions.
        faces: (F, 3) int triangle indices.
        target_faces: face count to aim for; a mesh already smaller is returned untouched.

    Returns:
        The decimated vertices and faces.
    """
    if len(faces) <= target_faces:
        return vertices, faces
    import fast_simplification
    reduction = 1.0 - float(target_faces) / float(len(faces))
    out_vertices, out_faces = fast_simplification.simplify(
        vertices.astype(np.float32), faces.astype(np.int32), reduction)
    return np.asarray(out_vertices, dtype=np.float64), np.asarray(out_faces, dtype=np.int64)


def transfer_attributes(source_vertices: np.ndarray, target_vertices: np.ndarray,
                        attributes: Dict[str, Optional[np.ndarray]]) -> Dict[str, Optional[np.ndarray]]:
    """
    Carry per-vertex attributes from the dense mesh onto a decimated one by nearest neighbour.

    Args:
        source_vertices: (V, 3) positions the attributes belong to.
        target_vertices: (W, 3) positions to attribute.
        attributes: name -> (V, C) array, or None to pass None through.

    Returns:
        The same keys, each resampled to (W, C).
    """
    from scipy.spatial import cKDTree
    _, index = cKDTree(source_vertices).query(target_vertices, k=1)
    return {key: (None if value is None else value[index]) for key, value in attributes.items()}


def convert(name: str, mesh_path: str, args: argparse.Namespace) -> Dict[str, object]:
    """
    Decimate, orient, normalise and write one object's glb.

    Args:
        name: object name.
        mesh_path: the ``mesh.ply`` to convert.
        args: the parsed command line.

    Returns:
        What was written, for the json index.
    """
    import trimesh
    mesh = trimesh.load(mesh_path, process=False)
    source_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    source_colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8) \
        if mesh.visual.kind == 'vertex' else None
    source_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    faces_in = len(mesh.faces)

    vertices, faces = decimate(source_vertices, np.asarray(mesh.faces), args.target_faces)
    resampled = transfer_attributes(source_vertices, vertices,
                                    {'colors': source_colors, 'normals': source_normals})
    colors, normals = resampled['colors'], resampled['normals']

    # The size of the object in the units the metrics are reported in, before it is normalised away.
    extents = (source_vertices.max(axis=0) - source_vertices.min(axis=0))

    rotation = upright_rotation(args.sfd_dir, args.data_root, name)
    if rotation is not None:
        vertices = vertices @ rotation.T
        normals = normals @ rotation.T

    vertices -= 0.5 * (vertices.max(axis=0) + vertices.min(axis=0))
    scale = float(np.abs(vertices).max())
    if scale > 0:
        vertices /= scale

    out = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors,
                          vertex_normals=normals, process=False)
    out_dir = os.path.join(args.output_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'mesh.glb')
    with open(out_path, 'wb') as handle:
        handle.write(trimesh.exchange.gltf.export_glb(out, include_normals=True))

    entry = {
        'name': name,
        'glb': 'assets/{}/mesh.glb'.format(name),
        'source': os.path.relpath(mesh_path, os.path.dirname(args.exps_dir.rstrip('/'))),
        'faces_source': faces_in,
        'faces': int(len(faces)),
        'vertices': int(len(vertices)),
        'bytes': os.path.getsize(out_path),
        'bytes_source': os.path.getsize(mesh_path),
        'upright': rotation is not None,
        'extent': [round(float(v), 4) for v in extents],
    }
    print('  {:9s} {:>9} -> {:>6} faces, {:>7.1f} KiB glb{}'.format(
        name, faces_in, entry['faces'], entry['bytes'] / 1024.0,
        '' if entry['upright'] else '  (no ground truth for "up", left in the canonical frame)'))
    return entry


def main() -> None:
    """Convert every object's exported mesh and write the json index."""
    args = parse_args()
    print('decimating to ~{} faces'.format(args.target_faces))
    entries: List[Dict[str, object]] = []
    for name in args.objects:
        mesh_path = latest_mesh(args.exps_dir, name)
        if mesh_path is None:
            print('  {:9s} no mesh export under {}/Mat-{}-mesh, skipped'.format(
                name, args.exps_dir, name))
            continue
        entries.append(convert(name, mesh_path, args))

    with open(args.index, 'w') as handle:
        json.dump(entries, handle, indent=2)
    total = sum(int(entry['bytes']) for entry in entries)
    print('wrote {} meshes, {:.1f} MiB total, index in {}'.format(
        len(entries), total / 1024.0 ** 2, args.index))


if __name__ == '__main__':
    main()
