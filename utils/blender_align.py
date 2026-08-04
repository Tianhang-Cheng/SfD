"""
Relate the SfM/network frames of a processed object to the Blender frame it was rendered in.

Everything the training code sees lives in the *SfM world* frame that COLMAP reconstructed
(``transforms_train.json``, ``object_pred_pose.json``, ``object_scale_matrix.json``); the
Blender ground truth of the synthetic objects lives in a different frame, related to it by
an unknown similarity: COLMAP fixes the scale arbitrarily.

For a synthetic object we know one camera in both frames, so

    R_s = R_blender_cam @ R_sfm_cam^T
    S(s) = [[s * R_s, t_blender_cam - s * R_s @ t_sfm_cam], [0, 1]]

maps the SfM world into the Blender world up to the single unknown ``s`` (both cameras use
the same OpenGL/Blender convention, see ``utils/rend_util.get_camera_params``). ``s`` is then
pinned down by the object poses: for every instance ``i``

    T_i = inv(M_i) @ S(s) @ O_i @ scale_mat

maps the canonical (network) space to the *object local* space of the Blender object, and
because all instances are copies of the same object, ``T_i`` must not depend on ``i``. The
translation of ``T_i`` is affine in ``s``, so the ``s`` that minimises the spread across
instances has a closed form (:func:`solve_scale_from_object_poses`).

``T`` is what makes a metric 3D comparison against the Blender mesh possible; the spread it
leaves over is a direct measure of how good the SfM poses are.
"""

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def load_json(path: str) -> Dict[str, Any]:
    """
    Read a json file.

    Args:
        path: file to read.

    Returns:
        The parsed content.
    """
    with open(path, 'r') as handle:
        return json.load(handle)


def load_object_poses(path: str, indexes: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    Read a ``*_object_*_pose.json`` written by the preprocessing or by Blender.

    Both files map ``'<object name>_<NN>'`` to a 4x4 object-to-world matrix. The predicted
    poses are indexed by position in ``non_empty_indexes.txt`` while the Blender ones are
    indexed by the original instance id, which is what ``indexes`` selects.

    Args:
        path: file to read.
        indexes: instance ids to pick, in order; None reads ``00..N-1``.

    Returns:
        ``[n,4,4]`` float64 array of object-to-world matrices.
    """
    meta = load_json(path)
    key_word = list(meta)[0].rsplit('_', 1)[0]
    if indexes is None:
        indexes = range(len(meta))
    return np.stack([np.array(meta['{}_{}'.format(key_word, str(int(i)).zfill(2))],
                              dtype=np.float64) for i in indexes], axis=0)


def matrix_scale(matrices: np.ndarray) -> np.ndarray:
    """
    Per axis scale baked into the rotation columns of a stack of transforms.

    Args:
        matrices: ``[n,4,4]`` array.

    Returns:
        ``[n,3]`` array of column norms.
    """
    return np.linalg.norm(matrices[:, :3, :3], axis=1)


def rotation_angle(rotations: np.ndarray) -> np.ndarray:
    """
    Geodesic angle of a stack of rotation matrices, in degrees.

    Args:
        rotations: ``[n,3,3]`` array of rotations (must be orthonormal, see
            :func:`polar_rotations`).

    Returns:
        ``[n]`` array of angles in degrees.
    """
    trace = np.trace(np.asarray(rotations, dtype=np.float64), axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def polar_rotations(matrices: np.ndarray) -> np.ndarray:
    """
    Rotation factor of the linear part of a stack of affine transforms.

    Some Blender objects carry an anisotropic object scale, so the transforms here are not
    similarities in general; the polar factor is still the well defined "rotation part".

    Args:
        matrices: ``[n,4,4]`` or ``[n,3,3]`` array.

    Returns:
        ``[n,3,3]`` float64 array of rotations.
    """
    linear = np.asarray(matrices, dtype=np.float64)[:, :3, :3]
    u, _, vh = np.linalg.svd(linear)
    rotation = u @ vh
    flip = np.linalg.det(rotation) < 0
    if flip.any():  # keep it a proper rotation for mirrored instances
        u[flip, :, -1] *= -1
        rotation = u @ vh
    return rotation


def cube_corners(radius: float = 1.0) -> np.ndarray:
    """
    The eight corners of the cube ``[-radius, radius]^3``, used as a probe set.

    Args:
        radius: half side length of the cube.

    Returns:
        ``[8,3]`` float64 array of corners.
    """
    return radius * np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1], indexing='ij'),
                             dtype=np.float64).reshape(3, -1).T


def transform_spread(per_instance: np.ndarray, average: np.ndarray,
                     radius: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    How far the per instance transforms disagree, measured on the canonical bounding cube.

    Args:
        per_instance: ``[n,4,4]`` transforms, one per instance.
        average: ``[4,4]`` the transform they should all be equal to.
        radius: half side length of the canonical cube to probe with.

    Returns:
        spread: ``[n]`` largest corner displacement per instance, in target units.
        size: the bounding box diagonal of the mapped cube, i.e. the scale the spread should
            be judged against.
    """
    corners = cube_corners(radius)
    mapped = np.einsum('nij,cj->nci', per_instance[:, :3, :3], corners) \
        + per_instance[:, None, :3, 3]
    reference = corners @ average[:3, :3].T + average[None, :3, 3]
    size = float(np.linalg.norm(reference.max(axis=0) - reference.min(axis=0)))
    return np.linalg.norm(mapped - reference[None], axis=2).max(axis=1), size


def similarity_from_cameras(sfm_camera: np.ndarray, blender_camera: np.ndarray,
                            scale: float) -> np.ndarray:
    """
    Build the SfM-world to Blender-world similarity for a given SfM scale.

    Args:
        sfm_camera: ``[4,4]`` camera-to-world of the training view in the SfM frame.
        blender_camera: ``[4,4]`` camera-to-world of the same view in the Blender frame.
        scale: the unknown SfM scale ``s``.

    Returns:
        ``[4,4]`` float64 similarity matrix mapping SfM world points to Blender world points.
    """
    rotation = blender_camera[:3, :3] @ sfm_camera[:3, :3].T
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = blender_camera[:3, 3] - scale * rotation @ sfm_camera[:3, 3]
    return matrix


def solve_scale_from_object_poses(sfm_camera: np.ndarray,
                                 blender_camera: np.ndarray,
                                 sfm_object_poses: np.ndarray,
                                 blender_object_poses: np.ndarray,
                                 scale_matrix: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Closed form solve of the SfM scale ``s``.

    ``T_i(s) = inv(M_i) @ S(s) @ O_i @ scale_mat`` has a translation that is affine in ``s``,
    ``t_i(s) = a_i + s * b_i``, so minimising the spread of the translations across instances
    is a scalar least squares problem.

    Args:
        sfm_camera: ``[4,4]`` camera-to-world in the SfM frame.
        blender_camera: ``[4,4]`` camera-to-world in the Blender frame.
        sfm_object_poses: ``[n,4,4]`` predicted object-to-world matrices ``O_i``.
        blender_object_poses: ``[n,4,4]`` Blender object-to-world matrices ``M_i``.
        scale_matrix: ``[4,4]`` canonical-to-SfM-world matrix from
            ``object_scale_matrix.json``.

    Returns:
        scale: the least squares ``s``.
        translations: ``[n,3]`` the resulting per instance translations ``t_i(s)``.
    """
    rotation = blender_camera[:3, :3] @ sfm_camera[:3, :3].T
    canonical_to_world = sfm_object_poses @ scale_matrix[None]
    inverse_blender = np.linalg.inv(blender_object_poses)

    offset = np.einsum('nij,nj->ni', inverse_blender[:, :3, :3],
                       blender_camera[None, :3, 3] - blender_object_poses[:, :3, 3])
    direction = np.einsum('nij,jk,nk->ni', inverse_blender[:, :3, :3], rotation,
                          canonical_to_world[:, :3, 3] - sfm_camera[None, :3, 3])

    centred_offset = offset - offset.mean(axis=0, keepdims=True)
    centred_direction = direction - direction.mean(axis=0, keepdims=True)
    denominator = float((centred_direction ** 2).sum())
    scale = 1.0 if denominator < 1e-20 else \
        float(-(centred_offset * centred_direction).sum() / denominator)
    return scale, offset + scale * direction


def solve_canonical_to_blender(sfm_camera: np.ndarray,
                               blender_camera: np.ndarray,
                               sfm_object_poses: np.ndarray,
                               blender_object_poses: np.ndarray,
                               scale_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Solve the canonical-to-Blender-object-local transform and report how well it fits.

    Args:
        sfm_camera: ``[4,4]`` camera-to-world in the SfM frame.
        blender_camera: ``[4,4]`` camera-to-world in the Blender frame.
        sfm_object_poses: ``[n,4,4]`` predicted object-to-world matrices ``O_i``.
        blender_object_poses: ``[n,4,4]`` Blender object-to-world matrices ``M_i``.
        scale_matrix: ``[4,4]`` canonical-to-SfM-world matrix.

    Returns:
        A dict with
            ``sfm_scale``: the recovered SfM scale ``s``,
            ``world_to_blender``: ``[4,4]`` SfM world -> Blender world,
            ``canonical_to_blender``: ``[4,4]`` the instance averaged ``T``,
            ``canonical_to_blender_per_instance``: ``[n,4,4]`` the individual ``T_i``,
            ``canonical_to_blender_world``: ``[n,4,4]`` canonical -> Blender *world* per
                instance, ``S @ O_i @ scale_mat``, which is a true similarity,
            ``canonical_scale_xyz``: singular values of ``T`` (canonical -> Blender unit),
            ``rotation_spread_deg``: ``[n]`` angle between the rotation part of each ``T_i``
                and of ``T``,
            ``corner_spread``: ``[n]`` largest disagreement of ``T_i`` with ``T`` over the
                canonical bounding cube, in Blender local units,
            ``object_size``: bounding box diagonal of the canonical cube mapped by ``T``,
            ``translation_spread``: ``[n]`` distance of each ``t_i`` from their mean,
            ``blender_pose_scale``: ``[n,3]`` scale baked into the Blender poses (anisotropic
                for some objects, which is why ``T`` is affine and not a similarity).
    """
    scale, _ = solve_scale_from_object_poses(sfm_camera, blender_camera, sfm_object_poses,
                                             blender_object_poses, scale_matrix)
    world_to_blender = similarity_from_cameras(sfm_camera, blender_camera, scale)
    canonical_to_world = world_to_blender[None] @ sfm_object_poses @ scale_matrix[None]
    per_instance = np.linalg.inv(blender_object_poses) @ canonical_to_world

    # the T_i are copies of the same transform up to pose noise, so the plain arithmetic mean
    # is the right estimator here (and unlike an SVD average it survives anisotropic scale)
    average = per_instance.mean(axis=0)
    spread, size = transform_spread(per_instance, average)
    translations = per_instance[:, :3, 3]
    return {
        'sfm_scale': scale,
        'world_to_blender': world_to_blender,
        'canonical_to_blender': average,
        'canonical_to_blender_per_instance': per_instance,
        'canonical_to_blender_world': canonical_to_world,
        'canonical_scale_xyz': np.linalg.svd(average[:3, :3], compute_uv=False),
        'rotation_spread_deg': rotation_angle(
            np.linalg.inv(polar_rotations(average[None])) @ polar_rotations(per_instance)),
        'corner_spread': spread,
        'object_size': size,
        'translation_spread': np.linalg.norm(
            translations - translations.mean(axis=0, keepdims=True), axis=1),
        'blender_pose_scale': matrix_scale(blender_object_poses),
    }


def average_transforms(matrices: np.ndarray) -> np.ndarray:
    """
    Average a stack of *similarity* transforms (rotation via SVD, scale and translation
    arithmetically).

    Only use this when the transforms are known to be similarities; the canonical-to-Blender
    transforms are not, because Blender objects may carry an anisotropic object scale.

    Args:
        matrices: ``[n,4,4]`` array of similarities.

    Returns:
        ``[4,4]`` float64 average.
    """
    scale = matrix_scale(matrices).mean()
    rotation = polar_rotations(matrices.mean(axis=0)[None])[0]
    average = np.eye(4, dtype=np.float64)
    average[:3, :3] = scale * rotation
    average[:3, 3] = matrices[:, :3, 3].mean(axis=0)
    return average


def solve_from_data_dir(data_split_dir: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Read a processed object directory and solve its alignment to the Blender ground truth.

    Args:
        data_split_dir: e.g. ``hf_data/train_split/coffee``, holding
            ``transforms_train.json``, ``blender_camera_gt_pose.json``,
            ``object_pred_pose.json``, ``blender_object_gt_pose.json``,
            ``object_scale_matrix.json`` and ``non_empty_indexes.txt``.
        verbose: print the recovered scale and the residual spread.

    Returns:
        The dict of :func:`solve_canonical_to_blender`, plus ``focal``, ``blender_focal``,
        ``non_empty_indexes``, ``sfm_camera``, ``blender_camera`` and ``scale_matrix``.

    Raises:
        FileNotFoundError: if the object has no Blender ground truth (a real world capture).
    """
    needed = ['transforms_train.json', 'blender_camera_gt_pose.json', 'object_pred_pose.json',
              'blender_object_gt_pose.json', 'object_scale_matrix.json']
    for name in needed:
        if not os.path.exists(os.path.join(data_split_dir, name)):
            raise FileNotFoundError(
                '{} has no {}: the Blender alignment only exists for the synthetic '
                'objects'.format(data_split_dir, name))

    indexes = np.atleast_1d(np.loadtxt(
        os.path.join(data_split_dir, 'non_empty_indexes.txt'))).astype(int)

    sfm_meta = load_json(os.path.join(data_split_dir, 'transforms_train.json'))
    blender_meta = load_json(os.path.join(data_split_dir, 'blender_camera_gt_pose.json'))
    sfm_camera = np.array(sfm_meta['frames'][0]['transform_matrix'], dtype=np.float64)
    blender_camera = np.array(blender_meta['frames'][0]['transform_matrix'], dtype=np.float64)

    sfm_object_poses = load_object_poses(os.path.join(data_split_dir, 'object_pred_pose.json'))
    blender_object_poses = load_object_poses(
        os.path.join(data_split_dir, 'blender_object_gt_pose.json'), indexes=indexes)
    scale_matrix = np.array(load_json(
        os.path.join(data_split_dir, 'object_scale_matrix.json'))['scale_matrix'],
        dtype=np.float64)

    result = solve_canonical_to_blender(sfm_camera, blender_camera, sfm_object_poses,
                                       blender_object_poses, scale_matrix)
    result['non_empty_indexes'] = indexes
    result['sfm_camera'] = sfm_camera
    result['blender_camera'] = blender_camera
    result['scale_matrix'] = scale_matrix
    result['focal'] = float(sfm_meta['focal'])
    result['blender_focal'] = focal_from_camera_angle(
        float(blender_meta['camera_angle_x']), int(blender_meta.get('resolution', 800)))

    if verbose:
        print('{}: {} instances'.format(data_split_dir, len(indexes)))
        print('  sfm scale s                = {:.5f}'.format(result['sfm_scale']))
        print('  canonical -> blender scale = {} (singular values)'.format(
            np.round(result['canonical_scale_xyz'], 5)))
        print('  rotation spread (deg)      mean {:.3f}, max {:.3f}'.format(
            float(result['rotation_spread_deg'].mean()),
            float(result['rotation_spread_deg'].max())))
        print('  corner spread              mean {:.5f}, max {:.5f} ({:.2%} of the object '
              'size {:.4f})'.format(float(result['corner_spread'].mean()),
                                    float(result['corner_spread'].max()),
                                    float(result['corner_spread'].max() / result['object_size']),
                                    result['object_size']))
    return result


def focal_from_camera_angle(camera_angle_x: float, resolution: int) -> float:
    """
    Convert Blender's horizontal field of view into a focal length in pixels.

    Args:
        camera_angle_x: horizontal field of view in radians.
        resolution: width of the render in pixels.

    Returns:
        The focal length in pixels.
    """
    return 0.5 * resolution / np.tan(0.5 * camera_angle_x)


def project_points_to_pixels(points: np.ndarray, camera_to_world: np.ndarray, focal: float,
                             resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project world points with the pinhole camera convention used throughout the repo.

    The cameras are stored the way Blender/OpenGL writes them: x right, y up, -z forward,
    which is what ``utils/rend_util.get_camera_params`` undoes with ``[1,-1,-1]``.

    Args:
        points: ``[n,3]`` array of world points.
        camera_to_world: ``[4,4]`` camera-to-world matrix in the same frame.
        focal: focal length in pixels (square pixels, principal point at the image centre).
        resolution: side length of the square image in pixels.

    Returns:
        pixels: ``[n,2]`` float array of ``(column, row)`` coordinates.
        depth: ``[n]`` distance along the viewing direction; negative means behind the camera.
    """
    world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
    camera_points = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3][None]
    depth = -camera_points[:, 2]
    safe_depth = np.where(np.abs(depth) < 1e-8, 1e-8, depth)
    centre = resolution / 2.0
    column = centre + focal * camera_points[:, 0] / safe_depth
    row = centre - focal * camera_points[:, 1] / safe_depth
    return np.stack([column, row], axis=-1), depth
