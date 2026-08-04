"""
Push the local dataset fixes in ``hf_data/`` back to the Hugging Face dataset repo.

Maintainer tool, not needed to train or evaluate anything. It uploads only the files listed in
:data:`FIXES`, checks each one before sending it, and prints what it would do with ``--dry_run``:

    python scripts/upload_dataset_fixes.py --dry_run
    python scripts/upload_dataset_fixes.py            # needs "hf auth login" or --token

Currently that is ``clock``'s ``blender_object_gt_pose.json``, which as released described a
different arrangement of the nine clocks than ``clock_clean.blend`` does; see
``scripts/blender_fix_gt_poses.py`` and the README section on comparing with the Blender ground
truth.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ID = 'TianhangCheng7/DuplicateSingleImage'

FIXES: List[Dict[str, str]] = [
    {
        'local': 'train_split/clock/blender_object_gt_pose.json',
        'remote': 'train_split/clock/blender_object_gt_pose.json',
        'check': 'object_poses',
        'why': "the released poses do not describe clock_clean.blend; regenerated from the scene "
               "by scripts/blender_fix_gt_poses.py",
    },
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse the command line.

    Args:
        argv: argument list to parse; ``sys.argv[1:]`` by default.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_root', type=str, default='hf_data',
                        help='local copy of the dataset the fixed files are taken from')
    parser.add_argument('--repo', type=str, default=REPO_ID,
                        help='dataset repository on the Hub to upload to')
    parser.add_argument('--token', type=str, default='',
                        help='Hub write token; by default the cached "hf auth login" token or '
                             'HF_TOKEN is used')
    parser.add_argument('--message', type=str, default='fix clock/blender_object_gt_pose.json',
                        help='commit message')
    parser.add_argument('--dry_run', default=False, action='store_true',
                        help='run the checks and print the plan without touching the Hub')
    return parser.parse_args(argv)


def check_object_poses(path: str) -> str:
    """
    Sanity check a ``blender_object_gt_pose.json`` before it is uploaded.

    Args:
        path: local path of the file.

    Returns:
        A one line description of what the file holds.

    Raises:
        ValueError: if the file is not a set of well formed object-to-world matrices.
    """
    with open(path) as handle:
        poses = json.load(handle)
    if not poses:
        raise ValueError('{}: no poses in this file'.format(path))
    matrices = np.array([np.asarray(poses[name], dtype=np.float64) for name in sorted(poses)])
    if matrices.shape[1:] != (4, 4):
        raise ValueError('{}: expected 4x4 matrices, got {}'.format(path, matrices.shape[1:]))
    if not np.isfinite(matrices).all():
        raise ValueError('{}: non-finite values in the matrices'.format(path))
    if not np.allclose(matrices[:, 3, :], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError('{}: bottom row is not [0, 0, 0, 1]'.format(path))
    determinants = np.linalg.det(matrices[:, :3, :3])
    if determinants.min() <= 0.0:
        raise ValueError('{}: mirrored pose, smallest determinant {:.3e}'
                         .format(path, determinants.min()))
    scales = np.linalg.norm(matrices[:, :3, :3], axis=1)
    return '{} poses, scale {:.4f}-{:.4f}, determinant {:.4f}-{:.4f}'.format(
        len(matrices), scales.min(), scales.max(), determinants.min(), determinants.max())


CHECKS = {'object_poses': check_object_poses}


def plan_uploads(data_root: str) -> List[Tuple[str, str, str]]:
    """
    Check every entry of :data:`FIXES` and return the ones that are ready to upload.

    Args:
        data_root: local copy of the dataset.

    Returns:
        One ``(local_path, path_in_repo, description)`` per file.

    Raises:
        FileNotFoundError: if a listed fix is missing locally.
        ValueError: if a listed fix does not pass its check.
    """
    uploads: List[Tuple[str, str, str]] = []
    for fix in FIXES:
        local = os.path.join(data_root, fix['local'])
        if not os.path.isfile(local):
            raise FileNotFoundError('{} is missing, so there is nothing to upload for {}'
                                    .format(local, fix['remote']))
        description = CHECKS[fix['check']](local)
        original = local + '.orig'
        if os.path.isfile(original):
            with open(local) as handle:
                fixed_text = handle.read()
            with open(original) as handle:
                original_text = handle.read()
            if fixed_text == original_text:
                raise ValueError('{} is identical to its .orig backup, so it was never fixed'
                                 .format(local))
            description += ', differs from the released file kept in .orig'
        uploads.append((local, fix['remote'], description))
    return uploads


def upload(uploads: List[Tuple[str, str, str]], repo: str, message: str,
           token: Optional[str]) -> None:
    """
    Upload the planned files to the dataset repository.

    Args:
        uploads: the plan returned by :func:`plan_uploads`.
        repo: dataset repository id on the Hub.
        message: commit message.
        token: write token, or None to use the cached login.
    """
    from huggingface_hub import HfApi

    api: Any = HfApi(token=token or None)
    for local, remote, _ in uploads:
        url = api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo,
                             repo_type='dataset', commit_message=message)
        print('uploaded {} -> {}'.format(local, url))


def main() -> None:
    """Check the local dataset fixes and push them to the Hub."""
    args = parse_args()
    uploads = plan_uploads(args.data_root)
    for fix, (local, remote, description) in zip(FIXES, uploads):
        print('{}\n  -> {}:{}\n  {}\n  {}'.format(local, args.repo, remote, description,
                                                  fix['why']))
    if args.dry_run:
        print('\n--dry_run, nothing uploaded')
        return
    upload(uploads, args.repo, args.message, args.token)


if __name__ == '__main__':
    main()
