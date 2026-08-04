"""Build synthetic stage-4 outputs so preprocess/5_sfm.py -> 7_*.py can be run offline.

SfD's stage 1-4 need the SuperPoint/SuperGlue checkpoints (Hugging Face) to produce
``final_feats.npz`` / ``final_matches.npz`` / ``image_pair.txt``.  This script fabricates those
three files from a known 3D object + known virtual camera poses, which is enough to exercise
stages 5, 6 and 7 -- the stages where COLMAP runs and where partial registration is handled.

Usage:
    python preprocess/debug_synth_sfm.py --instance_dir DIR --instance_num N [--drop_instance K]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from PIL import Image

from utils.pose_transform import blender_to_opencv

TRAIN_RES: int = 800
FOCAL: float = 1111.0


def make_object_points(num_points: int, rng: np.random.Generator) -> np.ndarray:
    """Sample 3D points on the surface of a box, i.e. a fake 'object'.

    Args:
        num_points: number of 3D points to sample.
        rng: numpy random generator.

    Returns:
        Array of shape ``[num_points, 3]``.
    """
    points = rng.uniform(-1.0, 1.0, size=(num_points, 3))
    axis = rng.integers(0, 3, size=num_points)
    sign = rng.choice([-1.0, 1.0], size=num_points)
    points[np.arange(num_points), axis] = sign
    return points


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build an OpenCV-convention world-to-camera matrix.

    Args:
        eye: camera position in world space, shape ``[3]``.
        target: point the camera looks at, shape ``[3]``.
        up: approximate up direction, shape ``[3]``.

    Returns:
        World-to-camera matrix of shape ``[4, 4]``.
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    true_up = np.cross(forward, right)
    rotation = np.stack([right, true_up, forward], axis=0)
    world_to_cam = np.eye(4)
    world_to_cam[:3, :3] = rotation
    world_to_cam[:3, 3] = -rotation @ eye
    return world_to_cam


def make_cameras(instance_num: int, rng: np.random.Generator) -> np.ndarray:
    """Place one virtual camera per instance on a sphere around the object.

    Args:
        instance_num: number of instances / virtual cameras.
        rng: numpy random generator.

    Returns:
        Stacked world-to-camera matrices, shape ``[instance_num, 4, 4]``.
    """
    poses: List[np.ndarray] = []
    radius = 5.0
    for i in range(instance_num):
        azimuth = 2.0 * np.pi * i / instance_num
        elevation = np.deg2rad(20.0 + 20.0 * rng.uniform(-1.0, 1.0))
        eye = radius * np.array([
            np.cos(elevation) * np.cos(azimuth),
            np.sin(elevation),
            np.cos(elevation) * np.sin(azimuth),
        ])
        poses.append(look_at(eye, np.zeros(3), np.array([0.0, 1.0, 0.0])))
    return np.stack(poses, axis=0)


def project(points: np.ndarray, world_to_cam: np.ndarray, noise: float,
            rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points into one virtual camera.

    Args:
        points: world-space points, shape ``[m, 3]``.
        world_to_cam: world-to-camera matrix, shape ``[4, 4]``.
        noise: standard deviation of pixel noise added to the projections.
        rng: numpy random generator.

    Returns:
        Tuple of ``(pixels, visible)`` where ``pixels`` has shape ``[m, 2]`` and ``visible`` is a
        boolean mask of shape ``[m]`` marking points in front of the camera and inside the image.
    """
    cam = (world_to_cam[:3, :3] @ points.T).T + world_to_cam[:3, 3]
    in_front = cam[:, 2] > 1e-3
    depth = np.where(in_front, cam[:, 2], 1.0)
    principal = TRAIN_RES / 2.0 - 0.5
    pixels = np.stack([
        FOCAL * cam[:, 0] / depth + principal,
        FOCAL * cam[:, 1] / depth + principal,
    ], axis=1)
    pixels = pixels + rng.normal(scale=noise, size=pixels.shape)
    inside = ((pixels[:, 0] > 0) & (pixels[:, 0] < TRAIN_RES - 1) &
              (pixels[:, 1] > 0) & (pixels[:, 1] < TRAIN_RES - 1))
    return pixels, in_front & inside


def cameras_from_reference_poses(reference_dir: str, instance_num: int) -> np.ndarray:
    """Rebuild the virtual cameras of an already-preprocessed object.

    Inverts what ``7_extract_sfm_pose_and_visualize.py`` computes.  That stage derives the
    object-to-world poses from the virtual cameras as
    ``obj_pose_pred[i] = vc_pred[0] @ vc_pred_inv[i]`` with ``vc_pred[0]`` stored as the
    camera-to-world matrix in ``transforms_train.json``, so here

        vc_pred_inv[i] = inv(sfm_c) @ obj_pose_pred[i]                 (NeuS convention)
        world_to_cam[i] = blender_to_opencv(vc_pred_inv[i])            (COLMAP convention)

    Using the released poses instead of a made-up camera rig means the fabricated
    correspondences describe the real instance layout, so stages 5-7 can be checked against a
    reference and the resulting dataset is consistent with the real image.

    Args:
        reference_dir: directory holding ``object_pred_pose.json`` and ``transforms_train.json``.
        instance_num: number of instances expected in the reference poses.

    Returns:
        Stacked world-to-camera matrices in COLMAP convention, shape ``[instance_num, 4, 4]``.
    """
    with open(os.path.join(reference_dir, 'transforms_train.json'), 'r') as handle:
        transforms = json.load(handle)
    sfm_c = np.array(transforms['frames'][0]['transform_matrix'])

    with open(os.path.join(reference_dir, 'object_pred_pose.json'), 'r') as handle:
        obj_poses = json.load(handle)
    keys = sorted(obj_poses)
    assert len(keys) == instance_num, \
        '{} has {} object poses but instance_num is {}'.format(
            reference_dir, len(keys), instance_num)

    obj_to_world = np.stack([np.array(obj_poses[k]) for k in keys], axis=0)
    return blender_to_opencv(np.linalg.inv(sfm_c)[None] @ obj_to_world)


def object_center_from_cameras(world_to_cam: np.ndarray) -> Tuple[np.ndarray, float]:
    """Estimate what the virtual cameras are looking at.

    Solves for the world point closest to every camera's optical axis in the least-squares
    sense, which is where the shared canonical object sits.

    Args:
        world_to_cam: stacked world-to-camera matrices, shape ``[n, 4, 4]``.

    Returns:
        Tuple of ``(center, radius)``: the estimated object center in world space, shape
        ``[3]``, and a plausible object radius derived from the camera distances.
    """
    rotations = world_to_cam[:, :3, :3]
    eyes = -np.einsum('nji,nj->ni', rotations, world_to_cam[:, :3, 3])  # R^T @ (-t)
    axes = rotations[:, 2, :]  # camera +z in world space

    # sum_i (I - d_i d_i^T) x = sum_i (I - d_i d_i^T) o_i
    lhs = np.zeros((3, 3))
    rhs = np.zeros(3)
    for eye, axis in zip(eyes, axes):
        projector = np.eye(3) - np.outer(axis, axis)
        lhs += projector
        rhs += projector @ eye
    center = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    radius = 0.15 * float(np.mean(np.linalg.norm(eyes - center, axis=1)))
    return center, radius


def main() -> None:
    """Write synthetic ``sfm_inputs`` for one instance folder."""
    global FOCAL

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_dir', type=str, required=True)
    parser.add_argument('--instance_num', type=int, required=True)
    parser.add_argument('--num_points', type=int, default=400)
    parser.add_argument('--noise', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--from_poses', type=str, default=None,
        help='directory of an already-preprocessed object (with object_pred_pose.json and '
             'transforms_train.json). Its virtual cameras are reused instead of a made-up '
             'camera rig, so the recovered poses can be compared against it')
    parser.add_argument(
        '--focal', type=float, default=None,
        help='focal length used to project the synthetic points (default: {} or, with '
             '--from_poses, the focal in its transforms_train.json)'.format(FOCAL))
    parser.add_argument(
        '--drop_instance', type=int, default=-1,
        help='simulate an instance COLMAP cannot register: keep its features but delete every '
             'pair that involves it')
    parser.add_argument(
        '--drop_instance_feats', action='store_true',
        help='also drop the instance from final_feats.npz, i.e. simulate stage 4 finding no good '
             'pair at all for it')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    instance_num = args.instance_num

    if args.focal is not None:
        FOCAL = args.focal
    elif args.from_poses is not None:
        with open(os.path.join(args.from_poses, 'transforms_train.json'), 'r') as handle:
            FOCAL = float(json.load(handle)['focal'])

    raw_dir = os.path.join(args.instance_dir, 'raw')
    temp_dir = os.path.join(raw_dir, 'temp')
    resized_dir = os.path.join(temp_dir, 'resized')
    sfm_input_dir = os.path.join(temp_dir, 'sfm_inputs')
    for path in (resized_dir, sfm_input_dir):
        os.makedirs(path, exist_ok=True)

    if args.from_poses is not None:
        cameras = cameras_from_reference_poses(args.from_poses, instance_num)
        center, radius = object_center_from_cameras(cameras)
        points = center + radius * make_object_points(args.num_points, rng)
        print('reused the virtual cameras of {}; object center {}, radius {:.4f}, focal {:.2f}'
              .format(args.from_poses, np.round(center, 4), radius, FOCAL))
    else:
        cameras = make_cameras(instance_num, rng)
        points = make_object_points(args.num_points, rng)

    keypoints: Dict[int, np.ndarray] = {}
    global_to_local: Dict[int, Dict[int, int]] = {}
    for i in range(instance_num):
        pixels, visible = project(points, cameras[i], args.noise, rng)
        local_ids = np.where(visible)[0]
        keypoints[i] = pixels[local_ids].astype(np.float32)
        global_to_local[i] = {int(g): int(l) for l, g in enumerate(local_ids)}

    # a plain grey image per instance: COLMAP only needs the file to exist
    for i in range(instance_num):
        Image.fromarray(np.full((TRAIN_RES, TRAIN_RES, 3), 128, dtype=np.uint8)).save(
            os.path.join(resized_dir, '{}_rgb.png'.format(str(i).zfill(3))))

    out_feats: Dict[str, object] = {}
    for i in range(instance_num):
        if args.drop_instance_feats and i == args.drop_instance:
            continue
        out_feats[str(i)] = {
            'keypoints': keypoints[i],
            'keypoints_back': keypoints[i],
            'scores': np.ones(len(keypoints[i]), dtype=np.float32),
            'descriptors': np.zeros((256, len(keypoints[i])), dtype=np.float32),
        }

    out_matches: Dict[str, object] = {}
    pair_lines: List[str] = []
    for i in range(instance_num):
        for j in range(i + 1, instance_num):
            if args.drop_instance in (i, j):
                continue
            matches0 = np.full(len(keypoints[i]), -1, dtype=np.int32)
            for global_id, local_i in global_to_local[i].items():
                local_j = global_to_local[j].get(global_id, -1)
                if local_j >= 0:
                    matches0[local_i] = local_j
            out_matches['{}_{}'.format(i, j)] = {
                'mkpts0back': keypoints[i][matches0 > -1],
                'mkpts1back': keypoints[j][matches0[matches0 > -1]],
                'matches0': matches0,
                'match_confidence0': np.ones(len(matches0), dtype=np.float32),
            }
            pair_lines.append('{}_rgb.png {}_rgb.png\n'.format(
                str(i).zfill(3), str(j).zfill(3)))

    np.savez(os.path.join(sfm_input_dir, 'final_feats.npz'), **out_feats)
    np.savez(os.path.join(sfm_input_dir, 'final_matches.npz'), **out_matches)
    with open(os.path.join(sfm_input_dir, 'image_pair.txt'), 'w') as handle:
        handle.writelines(pair_lines)

    np.save(os.path.join(temp_dir, 'gt_world_to_cam.npy'), cameras)
    print('wrote {} feats, {} pairs to {}'.format(
        len(out_feats), len(pair_lines), sfm_input_dir))


if __name__ == '__main__':
    main()
