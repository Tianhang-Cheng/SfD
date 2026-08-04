"""
Export the ground truth mesh of a duplicated object out of its ``.blend``.

Must be run by Blender:

    blender --background blender_data/coffee/coffee_clean.blend \\
        --python scripts/blender_export_gt_mesh.py -- \\
        --data_split_dir hf_data/train_split/coffee --output hf_data/train_split/coffee/gt --world

Writes ``gt_mesh_local.ply`` -- one instance in the local frame its recorded ground truth pose
implies, which is the frame the network's canonical mesh is compared in -- and
``gt_mesh_meta.json`` with the matrices it was taken with. With ``--world`` it also writes
``gt_mesh_world.ply``, every instance placed in the Blender world, for a whole-pile comparison;
that one holds only the instances SfM registered (``non_empty_indexes.txt``), so that it contains
exactly what the network reconstructs, unless ``--include_unregistered`` says otherwise.

The released scenes do not name their objects ``<object>_00`` the way
``blender_object_gt_pose.json`` does, and five of them keep the whole pile joined into a single
mesh instead of one object per instance; :func:`scripts.blender_common.resolve_instances` sorts
both cases out and prints what it decided, and :func:`cut_pile` splits a joined pile back into
its instances.

Modifiers are evaluated through the dependency graph, so subdivision, mirrors and the like end
up in the exported triangles exactly as they were rendered. ``scripts/eval_mesh_3d.py`` then
compares an exported network mesh against these files.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blender_common import (JOINED, load_gt_poses, resolve_instances, split_into_instances,
                            write_ply)


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
    parser.add_argument('--output', type=str, required=True,
                        help='destination directory for the ply files and the metadata')
    parser.add_argument('--data_split_dir', type=str, required=True,
                        help='processed object directory; its blender_object_gt_pose.json '
                             'defines the instances to export')
    parser.add_argument('--instance', type=int, default=0,
                        help='index of the instance whose local frame gt_mesh_local.ply is '
                             'written in; the frames only differ by the ground truth pose, so '
                             'this normally does not matter')
    parser.add_argument('--world', default=False, action='store_true',
                        help='also export every instance placed in the Blender world frame')
    parser.add_argument('--include_unregistered', default=False, action='store_true',
                        help='keep the instances SfM failed to register; by default the export '
                             'is restricted to non_empty_indexes.txt, so that the world frame '
                             'ground truth holds exactly the instances the network reconstructs')
    parser.add_argument('--all_instances', default=False, action='store_true',
                        help='also export each instance separately in its own local frame, to '
                             'check that the duplicates really share one mesh')
    return parser.parse_args(argv)


def evaluate_mesh(object_name: str,
                  matrix: Optional[Sequence[Sequence[float]]] = None) -> Tuple[Any, Any]:
    """
    Evaluate an object's modifiers and return its triangulated mesh.

    Args:
        object_name: name of the object in ``bpy.data.objects``.
        matrix: 4x4 matrix to bake into the vertices; None keeps them in the object local frame.

    Returns:
        vertices: ``[n,3]`` float64 numpy array.
        faces: ``[m,3]`` int32 numpy array.

    Raises:
        KeyError: if the scene has no object with that name.
    """
    import bmesh
    import bpy
    import numpy as np
    from mathutils import Matrix

    if object_name not in bpy.data.objects:
        raise KeyError('no object named {!r} in this .blend'.format(object_name))
    source = bpy.data.objects[object_name]
    evaluated = source.evaluated_get(bpy.context.evaluated_depsgraph_get())

    mesh = bmesh.new()
    mesh.from_mesh(evaluated.to_mesh())
    if matrix is not None:
        mesh.transform(Matrix([list(row) for row in matrix]))
    bmesh.ops.triangulate(mesh, faces=mesh.faces)

    vertices = np.array([[v.co.x, v.co.y, v.co.z] for v in mesh.verts], dtype=np.float64)
    faces = np.array([[v.index for v in f.verts] for f in mesh.faces], dtype=np.int32)
    mesh.free()
    evaluated.to_mesh_clear()
    return vertices, faces


def export_world_pile(instances: Sequence[Dict[str, Any]], pieces: Dict[int, Tuple[Any, Any]],
                      output_dir: str) -> str:
    """
    Write the whole pile in Blender world coordinates.

    Args:
        instances: the instance records returned by
            :func:`scripts.blender_common.resolve_instances`.
        pieces: world space geometry per instance index, from :func:`cut_pile`; empty for the
            one-object-per-instance layout, where the geometry comes straight off the objects.
        output_dir: destination directory.

    Returns:
        The path that was written.
    """
    import numpy as np

    all_vertices: List[Any] = []
    all_faces: List[Any] = []
    for instance in instances:
        vertices, faces = instance_world_mesh(instance, pieces)
        all_faces.append(faces + sum(len(chunk) for chunk in all_vertices))
        all_vertices.append(vertices)
    path = os.path.join(output_dir, 'gt_mesh_world.ply')
    write_ply(path, np.concatenate(all_vertices, axis=0), np.concatenate(all_faces, axis=0))
    print('{} instances in the blender world frame -> {}'.format(len(instances), path))
    return path


def cut_pile(joined_object: str, poses: Dict[str, List[List[float]]]) -> Dict[int, Tuple[Any, Any]]:
    """
    Cut a joined pile into one world space mesh per instance.

    *Every* instance takes part in the split, including the ones SfM never registered, so that
    their geometry cannot end up attributed to a neighbour.

    Args:
        joined_object: name of the object holding the whole pile.
        poses: the full ground truth pose dictionary.

    Returns:
        World space ``(vertices, faces)`` per instance index.
    """
    import bpy
    import numpy as np

    matrix_world = [list(row) for row in bpy.data.objects[joined_object].matrix_world]
    vertices, faces = evaluate_mesh(joined_object, matrix=matrix_world)
    names = sorted(poses)
    matrices = np.array([np.asarray(poses[name], dtype=np.float64) for name in names])
    pieces, residual = split_into_instances(vertices, faces, matrices)
    print('  cut {} ({} vertices) into {}'.format(joined_object, len(vertices),
                                                  [len(piece[0]) for piece in pieces]))
    print('  recorded poses vs the poses in the scene: {:.2e}'.format(residual))
    if residual > 1e-3:
        print('  WARNING: blender_object_gt_pose.json does not describe this scene, so the '
              'instance labelling and the local frame are not trustworthy; the pile itself and '
              'gt_mesh_world.ply are still exact')
    return {index: piece for index, piece in enumerate(pieces)}


def instance_world_mesh(instance: Dict[str, Any],
                        pieces: Dict[int, Tuple[Any, Any]]) -> Tuple[Any, Any]:
    """
    Geometry of one instance in the Blender world frame, taken from the scene as it is saved.

    The scene is the authority here, not ``blender_object_gt_pose.json``: the recorded pose can
    differ from the object's own ``matrix_world`` by a constant local origin offset -- 0.32 Blender
    units on ``cash``, 0.33 on ``sign`` -- so placing the object with the recorded matrix would
    shift it by that much.

    Args:
        instance: one record from :func:`scripts.blender_common.resolve_instances`.
        pieces: world space geometry per instance index, from :func:`cut_pile`; empty for the
            one-object-per-instance layout.

    Returns:
        vertices: ``[n,3]`` float64 array in Blender world coordinates.
        faces: ``[m,3]`` int array.
    """
    import bpy

    if pieces:
        return pieces[instance['index']]
    matrix_world = [list(row) for row in bpy.data.objects[instance['object']].matrix_world]
    return evaluate_mesh(instance['object'], matrix=matrix_world)


def instance_local_mesh(instance: Dict[str, Any],
                        pieces: Dict[int, Tuple[Any, Any]]) -> Tuple[Any, Any]:
    """
    Geometry of one instance in the local frame its recorded ground truth pose implies.

    This is the frame ``utils/blender_align.py`` maps the canonical mesh into, so it has to be the
    frame of ``instance['matrix']`` and not the object's own frame -- the two differ whenever the
    recorded pose uses a different local origin.

    Args:
        instance: one record from :func:`scripts.blender_common.resolve_instances`.
        pieces: world space geometry per instance index, from :func:`cut_pile`; empty for the
            one-object-per-instance layout.

    Returns:
        vertices: ``[n,3]`` float64 array.
        faces: ``[m,3]`` int array.
    """
    import numpy as np

    vertices, faces = instance_world_mesh(instance, pieces)
    matrix = np.asarray(instance['matrix'], dtype=np.float64)
    return np.linalg.solve(matrix[:3, :3], (vertices - matrix[:3, 3]).T).T, faces


def main() -> None:
    """Export the ground truth geometry and the transforms it was taken with."""
    try:
        import bpy
        import numpy as np
    except ImportError:
        raise SystemExit('this needs Blender: either\n'
                         '  blender --background <object>.blend --python {} -- --help\n'
                         'or the pip module, which brings its own Blender:\n'
                         '  pip install bpy && python {} -- --blend_file <object>.blend --help'
                         .format(os.path.relpath(__file__), os.path.relpath(__file__)))
    args = parse_args()
    if args.blend_file:
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend_file))

    instances, layout = resolve_instances(args.data_split_dir, args.include_unregistered)
    pieces: Dict[int, Tuple[Any, Any]] = {}
    if layout == JOINED:
        pieces = cut_pile(instances[0]['object'], load_gt_poses(args.data_split_dir))
    os.makedirs(args.output, exist_ok=True)

    chosen = next((instance for instance in instances if instance['index'] == args.instance),
                  instances[0])
    vertices, faces = instance_local_mesh(chosen, pieces)
    local_path = os.path.join(args.output, 'gt_mesh_local.ply')
    write_ply(local_path, vertices, faces)
    print('{} ({}): {} vertices, {} faces -> {}'.format(chosen['name'], chosen['object'],
                                                        len(vertices), len(faces), local_path))

    meta: Dict[str, Any] = {
        'blend_file': bpy.data.filepath,
        'layout': layout,
        'local_instance': chosen['name'],
        'local_instance_index': chosen['index'],
        'local_matrix': chosen['matrix'],
        'bounding_box_local': [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        'instances': [{'name': instance['name'], 'index': instance['index'],
                       'object': instance['object'], 'matrix': instance['matrix']}
                      for instance in instances],
    }

    if args.world:
        export_world_pile(instances, pieces, args.output)

    if args.all_instances:
        boxes = []
        for instance in instances:
            instance_vertices, instance_faces = instance_local_mesh(instance, pieces)
            path = os.path.join(args.output,
                                'gt_mesh_local_{:02d}.ply'.format(instance['index']))
            write_ply(path, instance_vertices, instance_faces)
            boxes.append(np.concatenate([instance_vertices.min(axis=0),
                                         instance_vertices.max(axis=0)]))
            print('{} ({}) -> {} ({} vertices)'.format(instance['name'], instance['object'],
                                                       path, len(instance_vertices)))
        spread = float(np.abs(np.asarray(boxes) - np.mean(boxes, axis=0)).max())
        print('the instances agree to {:.3e} blender units on their local bounding box '
              '(they are duplicates, so this should be tiny)'.format(spread))
        meta['local_bounding_box_spread'] = spread

    meta_path = os.path.join(args.output, 'gt_mesh_meta.json')
    with open(meta_path, 'w') as handle:
        json.dump(meta, handle, indent=2)
    print('wrote {}'.format(meta_path))


if __name__ == '__main__':
    main()
