"""
Helpers shared by the scripts that run inside Blender.

Only the standard library and numpy are imported at module level: Blender ships its own Python
without torch, so nothing here may reach into ``utils/``.

The released ``.blend`` files do **not** name their objects the way
``blender_object_gt_pose.json`` does, and they come in two layouts. Both already hold the whole
pile exactly as it was rendered -- verified by rendering them and comparing with the training
images, see the README -- so nothing has to be moved or duplicated; the difference only matters
when a *single* instance has to be pulled back out:

* one object per instance -- ``box``, ``cash``, ``cleaner``, ``sign``. Objects are paired with the
  ground truth instances by their ``matrix_world`` (:func:`pair_by_matrix`); the names carry no
  usable order.
* the whole pile joined into one mesh -- ``clock``, ``coffee``, ``fire``, ``gitar``, ``tin``. The
  join left every instance as a contiguous run of vertex indices, so :func:`split_into_instances`
  cuts them apart exactly and matches the runs to the ground truth poses.
"""

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

ONE_OBJECT_PER_INSTANCE = 'one_object_per_instance'
JOINED = 'joined'


def load_gt_poses(data_split_dir: str) -> Dict[str, List[List[float]]]:
    """
    Read ``blender_object_gt_pose.json``.

    Args:
        data_split_dir: processed object directory.

    Returns:
        A mapping from instance name to its 4x4 ``matrix_world`` as nested lists.

    Raises:
        FileNotFoundError: if the file is not there.
    """
    path = os.path.join(data_split_dir, 'blender_object_gt_pose.json')
    with open(path, 'r') as handle:
        return json.load(handle)


def registered_indexes(data_split_dir: str) -> Optional[List[int]]:
    """
    Read the instance indexes SfM managed to register.

    Args:
        data_split_dir: processed object directory.

    Returns:
        The sorted indexes, or None if ``non_empty_indexes.txt`` does not exist.
    """
    path = os.path.join(data_split_dir, 'non_empty_indexes.txt')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as handle:
        return sorted(int(float(token)) for token in handle.read().split())


def scene_mesh_objects() -> List[str]:
    """
    Names of the mesh objects in the open ``.blend``, sorted, hidden ones included.

    Returns:
        The object names.
    """
    import bpy

    return sorted(obj.name for obj in bpy.data.objects if obj.type == 'MESH')


def vertex_count(object_name: str) -> int:
    """
    Number of vertices of an object's mesh data, before modifiers.

    Args:
        object_name: name of the object in ``bpy.data.objects``.

    Returns:
        The vertex count, or 0 if the object holds no mesh.
    """
    import bpy

    data = bpy.data.objects[object_name].data
    return len(data.vertices) if data is not None and hasattr(data, 'vertices') else 0


def matrix_deviation(object_name: str, matrix: Sequence[Sequence[float]]) -> float:
    """
    Largest absolute difference between an object's ``matrix_world`` and a ground truth matrix.

    Args:
        object_name: name of the object in ``bpy.data.objects``.
        matrix: 4x4 matrix as nested sequences.

    Returns:
        The maximum absolute element-wise difference.
    """
    import bpy
    import numpy as np

    current = np.array([list(row) for row in bpy.data.objects[object_name].matrix_world],
                       dtype=np.float64)
    return float(np.abs(current - np.asarray(matrix, dtype=np.float64)).max())


def contiguous_blocks(num_vertices: int, faces: Any, count: int) -> Any:
    """
    Cut a joined mesh into equal contiguous blocks of vertex indices, one per instance.

    Blender's *join* appends the objects one after another and never renumbers, so a pile of
    duplicates joined into one mesh keeps every instance as a contiguous run of ``num_vertices //
    count`` indices. That makes the split exact, which a geometric rule cannot be: assigning
    connected parts to their closest instance origin mixes up the interpenetrating duplicates in
    ``clock``, ``fire`` and ``gitar``.

    Args:
        num_vertices: number of vertices in the joined mesh.
        faces: ``[m,k]`` int array of vertex indices, used only to check the result.
        count: number of instances the mesh holds.

    Returns:
        ``[num_vertices]`` int array of block indices.

    Raises:
        ValueError: if the vertex count is not a multiple of ``count``, or if a face straddles a
            block boundary -- either way the mesh is not a plain join of ``count`` duplicates and
            this rule does not apply to it.
    """
    import numpy as np

    block, remainder = divmod(num_vertices, count)
    if remainder:
        raise ValueError('{} vertices do not split into {} equal blocks, so this mesh is not a '
                         'plain join of {} duplicates'.format(num_vertices, count, count))
    block_of = np.arange(num_vertices) // block
    faces = np.asarray(faces)
    straddling = int((block_of[faces].min(axis=1) != block_of[faces].max(axis=1)).sum())
    if straddling:
        raise ValueError('{} face(s) straddle a block boundary, so the instances are not '
                         'contiguous runs of vertex indices'.format(straddling))
    return block_of


def rigid_transform(source: Any, target: Any) -> Any:
    """
    Rigid transform mapping ``source`` onto ``target``, for two point sets in matched order.

    Args:
        source: ``[n,3]`` points.
        target: ``[n,3]`` points, the i-th corresponding to the i-th of ``source``.

    Returns:
        The ``[4,4]`` transform, reflection-free.
    """
    import numpy as np

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_centre, target_centre = source.mean(axis=0), target.mean(axis=0)
    left, _, right = np.linalg.svd((target - target_centre).T @ (source - source_centre))
    rotation = left @ np.diag([1.0, 1.0, np.sign(np.linalg.det(left @ right))]) @ right
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3], matrix[:3, 3] = rotation, target_centre - rotation @ source_centre
    return matrix


def match_blocks_to_instances(blocks: Sequence[Any], matrices: Any) -> Tuple[Any, float]:
    """
    Work out which block of a joined pile belongs to which ground truth instance.

    The blocks come out in the order the objects were joined, which has nothing to do with the
    order of ``blender_object_gt_pose.json``, and matching them by "closest origin" is not safe --
    the recorded origin can sit 0.34 Blender units from the geometry on ``sign``, and the
    interpenetrating piles put a block nearer a neighbour's origin than its own.

    Because the blocks are duplicates joined in one go they share their vertex order, so the exact
    rigid transform from block 0 to block ``b`` follows from the matched point sets. Anchoring
    block 0 on instance ``a`` then *predicts* the full pose of every other block, and the
    prediction is matched against the recorded poses as a permutation. Every anchor is tried and
    the best one wins, so the returned residual is a real measurement: how far the recorded poses
    are from the geometry they are supposed to describe.

    Args:
        blocks: ``k`` arrays of ``[n,3]`` world space vertices, in matched vertex order.
        matrices: ``[k,4,4]`` ground truth poses, in instance order.

    Returns:
        assignment: ``[k]`` int array giving the instance index of each block.
        residual: largest ``|predicted - recorded|`` matrix element over the chosen pairs, in
            Blender units. ~1e-7 means the recorded poses *are* the poses in the scene.
    """
    import numpy as np

    matrices = np.asarray(matrices, dtype=np.float64)
    relative = np.array([rigid_transform(blocks[0], block) for block in blocks])

    best: Optional[Tuple[Any, float]] = None
    for anchor in range(len(matrices)):
        predicted = relative @ matrices[anchor]
        cost = np.abs(predicted[:, None] - matrices[None]).reshape(len(matrices),
                                                                  len(matrices), -1).max(axis=2)
        assignment = balanced_assignment(cost, 1)
        residual = float(cost[np.arange(len(cost)), assignment].max())
        if best is None or residual < best[1]:
            best = (assignment, residual)
    return best


def balanced_assignment(cost: Any, capacity: int) -> Any:
    """
    Assign rows to columns cheapest-first, giving every column the same number of rows.

    Args:
        cost: ``[n,k]`` array of assignment costs.
        capacity: how many rows each column takes; ``n`` must be ``capacity * k``.

    Returns:
        ``[n]`` int array of column indices.
    """
    import numpy as np

    cost = np.asarray(cost, dtype=np.float64)
    assignment = np.full(len(cost), -1, dtype=np.int64)
    used = np.zeros(cost.shape[1], dtype=np.int64)
    pairs = np.stack(np.unravel_index(np.argsort(cost, axis=None), cost.shape), axis=1)
    for row, column in pairs:
        if assignment[row] < 0 and used[column] < capacity:
            assignment[row] = column
            used[column] += 1
    return assignment


def split_into_instances(vertices: Any, faces: Any, matrices: Any) -> Tuple[List[Tuple[Any, Any]],
                                                                           float]:
    """
    Cut a joined pile into one world space mesh per instance.

    :func:`contiguous_blocks` does the cutting and :func:`match_blocks_to_instances` decides which
    block is which instance. The pieces come out shape-identical on all five joined scenes -- their
    per-instance local bounding boxes agree to ~1e-7 Blender units -- so the cut itself is exact
    rather than a heuristic; the returned residual says whether the *labelling* can be trusted too.

    Args:
        vertices: ``[n,3]`` array of vertices in **world** coordinates.
        faces: ``[m,3]`` int array of vertex indices.
        matrices: ``[k,4,4]`` ground truth poses, in instance order.

    Returns:
        pieces: one ``(vertices, faces)`` pair per instance, in the order of ``matrices``, still in
            world coordinates and with the face indices remapped.
        residual: the match residual reported by :func:`match_blocks_to_instances`.
    """
    import numpy as np

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    matrices = np.asarray(matrices, dtype=np.float64)

    block_of = contiguous_blocks(len(vertices), faces, len(matrices))
    assignment, residual = match_blocks_to_instances(
        [vertices[block_of == block] for block in range(len(matrices))], matrices)

    pieces: List[Tuple[Any, Any]] = [(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))] * \
        len(matrices)
    for block, instance in enumerate(assignment):
        keep = block_of == block
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[keep] = np.arange(int(keep.sum()))
        pieces[instance] = (vertices[keep], remap[faces[keep[faces].all(axis=1)]])
    return pieces, residual


def pair_by_matrix(objects: Sequence[str],
                   poses: Dict[str, List[List[float]]]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Pair scene objects with ground truth instances by their ``matrix_world``.

    The object names of the released scenes carry no usable order -- on ``box``, pairing by sorted
    name puts instances 1.94 Blender units away from where the ground truth says they are -- so the
    matrices themselves have to do the matching. Instances are taken cheapest first and every
    object is used once, which is enough here because a pile of duplicates has one object per
    distinct pose.

    Args:
        objects: names of the mesh objects in the open ``.blend``.
        poses: the ground truth poses, keyed by instance name.

    Returns:
        paired: instance name -> object name.
        report: ``deviation`` (largest raw ``matrix_world`` difference over the chosen pairs),
            ``rotation_deviation`` (the same restricted to the linear part) and ``origin_offset`` /
            ``origin_offset_spread``. A pairing is right when the rotation deviation and the offset
            spread are both ~0; a nonzero but *constant* ``origin_offset`` only means the recorded
            poses use a different local origin than the objects do, which is harmless as long as
            the geometry is pulled back through the recorded matrix.
    """
    import numpy as np

    names = sorted(poses)
    cost = np.array([[matrix_deviation(obj, poses[name]) for obj in objects] for name in names])
    paired: Dict[str, str] = {}
    taken = set()
    for row in np.argsort(cost.min(axis=1)):
        order = [column for column in np.argsort(cost[row]) if column not in taken]
        taken.add(order[0])
        paired[names[row]] = objects[order[0]]

    import bpy

    rotation_deviation, offsets = 0.0, []
    for name in names:
        actual = np.array([list(row) for row in bpy.data.objects[paired[name]].matrix_world],
                          dtype=np.float64)
        recorded = np.asarray(poses[name], dtype=np.float64)
        rotation_deviation = max(rotation_deviation,
                                 float(np.abs(actual[:3, :3] - recorded[:3, :3]).max()))
        offsets.append(np.linalg.solve(actual[:3, :3], recorded[:3, 3] - actual[:3, 3]))
    offsets = np.asarray(offsets)
    report = {'deviation': float(max(matrix_deviation(paired[name], poses[name])
                                     for name in names)),
              'rotation_deviation': rotation_deviation,
              'origin_offset': offsets.mean(axis=0).tolist(),
              'origin_offset_spread': float(np.abs(offsets - offsets.mean(axis=0)).max())}
    return paired, report


def scene_layout(data_split_dir: str) -> Tuple[str, List[str], Dict[str, List[List[float]]]]:
    """
    Work out how the open ``.blend`` stores the pile.

    Args:
        data_split_dir: processed object directory holding ``blender_object_gt_pose.json``.

    Returns:
        layout: :data:`ONE_OBJECT_PER_INSTANCE` or :data:`JOINED`.
        objects: the sorted mesh object names in the scene.
        poses: the ground truth poses, keyed by instance name.
    """
    poses = load_gt_poses(data_split_dir)
    objects = scene_mesh_objects()
    layout = ONE_OBJECT_PER_INSTANCE if len(objects) == len(poses) else JOINED
    print('{} mesh object(s) in the scene, {} ground truth instances -> {}'
          .format(len(objects), len(poses), layout))
    return layout, objects, poses


def resolve_instances(data_split_dir: str, include_unregistered: bool = False,
                      tolerance: float = 1e-3) -> Tuple[List[Dict[str, Any]], str]:
    """
    Pair every ground truth instance with the object that holds its geometry.

    Args:
        data_split_dir: processed object directory holding ``blender_object_gt_pose.json``.
        include_unregistered: keep instances that are missing from ``non_empty_indexes.txt``,
            i.e. the ones SfM never registered and the network therefore cannot reconstruct.
        tolerance: largest ``matrix_world`` deviation, in Blender units, still counted as "this
            scene object really is that ground truth instance".

    Returns:
        instances: one dict per kept instance with ``name`` (the ground truth key), ``index``
            (its position in the sorted ground truth keys), ``matrix`` (4x4 nested lists) and
            ``object`` (the scene object holding its geometry; the joined pile for every instance
            in the ``JOINED`` layout).
        layout: which layout was detected.
    """
    layout, objects, poses = scene_layout(data_split_dir)
    import numpy as np

    names = sorted(poses)
    keep = registered_indexes(data_split_dir)
    if keep is not None and not include_unregistered:
        dropped = [name for index, name in enumerate(names) if index not in keep]
        if dropped:
            print('skipping the unregistered instances {} (--include_unregistered keeps them)'
                  .format(dropped))
        names = [name for index, name in enumerate(names) if index in keep]

    if layout == ONE_OBJECT_PER_INSTANCE:
        paired, report = pair_by_matrix(objects, poses)
        print('  paired object <-> instance by matrix_world: raw deviation {deviation:.2e}, '
              'rotation {rotation_deviation:.2e}'.format(**report))
        if report['origin_offset_spread'] <= tolerance and report['deviation'] > tolerance:
            print('  the recorded poses use a local origin offset by {} (constant to {:.1e}), so '
                  'the geometry is pulled back through the recorded matrix'
                  .format(np.round(report['origin_offset'], 5).tolist(),
                          report['origin_offset_spread']))
        if max(report['rotation_deviation'], report['origin_offset_spread']) > tolerance:
            print('  WARNING: above the {:.0e} tolerance, so at least one instance has no object '
                  'sitting on its ground truth pose'.format(tolerance))
    else:
        joined = max(objects, key=vertex_count) if objects else ''
        if not joined:
            raise ValueError('the scene holds no mesh object')
        paired = {name: joined for name in sorted(poses)}
        print('  the pile is joined into {!r}; single instances are cut out of it'.format(joined))

    instances = [{'name': name, 'index': sorted(poses).index(name), 'matrix': poses[name],
                  'object': paired[name]} for name in names]
    return instances, layout


def write_ply(path: str, vertices: Any, faces: Any) -> str:
    """
    Write a binary little endian PLY.

    This repeats :func:`utils.mesh_util.write_ply` on purpose: Blender's bundled Python has
    numpy but no torch, so ``utils.mesh_util`` cannot be imported here.

    Args:
        path: destination file.
        vertices: ``[n,3]`` array.
        faces: ``[m,3]`` int array.

    Returns:
        The path that was written.
    """
    import numpy as np

    vertices = np.asarray(vertices, dtype='<f4')
    faces = np.asarray(faces, dtype='<i4')
    header = ('ply\nformat binary_little_endian 1.0\n'
              'element vertex {}\nproperty float x\nproperty float y\nproperty float z\n'
              'element face {}\nproperty list uchar int vertex_indices\n'
              'end_header\n').format(len(vertices), len(faces))
    face_records = np.empty(len(faces), dtype=[('n', 'u1'), ('v', '<i4', (3,))])
    face_records['n'] = 3
    face_records['v'] = faces
    with open(path, 'wb') as handle:
        handle.write(header.encode('ascii'))
        handle.write(vertices.tobytes())
        handle.write(face_records.tobytes())
    return path
