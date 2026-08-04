"""Fetch the large binary assets that are not tracked in this git repository.

Only code lives in git. The pretrained SuperPoint/SuperGlue checkpoints and the HDRI
environment maps are hosted on the Hugging Face Hub and pulled on demand:

  * checkpoints   -> https://huggingface.co/TianhangCheng7/DuplicateWeight
  * envmaps (.exr)-> https://huggingface.co/datasets/TianhangCheng7/DuplicateBlenderData

Run once after cloning:

    python download_assets.py

or fetch a single group with --weights / --envmaps / --blender-data. The `ensure_*`
helpers below are also called lazily from the code that needs the files, so skipping
this step is not fatal.
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

WEIGHT_REPO = 'TianhangCheng7/DuplicateWeight'
BLENDER_REPO = 'TianhangCheng7/DuplicateBlenderData'

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / 'preprocess' / 'keypoint_matching' / 'weights'
ENVMAP_DIR = ROOT / 'envmaps'

# local file name -> path inside the Hugging Face repo
KEYPOINT_WEIGHTS = {
    'superpoint_v1.pth': 'keypoint_matching/superpoint_v1.pth',
    'superglue_indoor.pth': 'keypoint_matching/superglue_indoor.pth',
    'superglue_outdoor.pth': 'keypoint_matching/superglue_outdoor.pth',
}
ENVMAPS = {
    'b.exr': 'hdi/b.exr',
    'c.exr': 'hdi/c.exr',
    'd.exr': 'hdi/d.exr',
}


def _hf_hub_download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            'huggingface_hub is required to download the assets. '
            'Install it with "pip install -U huggingface_hub" '
            '(it is also listed in requirements.txt).'
        )
    return hf_hub_download


def _fetch(repo_id, repo_type, filename, dest, force=False):
    """Download `filename` from a Hub repo to `dest`, skipping if already present."""
    dest = Path(dest)
    if dest.exists() and not force:
        return dest

    hf_hub_download = _hf_hub_download()
    dest.parent.mkdir(parents=True, exist_ok=True)
    # stage in the destination directory so the final move never crosses filesystems
    tmp_dir = tempfile.mkdtemp(prefix='.hf_tmp_', dir=str(dest.parent))
    try:
        print('downloading {}:{} -> {}'.format(repo_id, filename, dest.relative_to(ROOT)))
        src = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            local_dir=tmp_dir,
        )
        os.replace(src, dest)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return dest


def ensure_keypoint_weights(force=False):
    """Make sure the SuperPoint/SuperGlue checkpoints are in preprocess/keypoint_matching/weights."""
    return [
        _fetch(WEIGHT_REPO, 'model', remote, WEIGHTS_DIR / name, force=force)
        for name, remote in KEYPOINT_WEIGHTS.items()
    ]


def ensure_envmap(name, force=False):
    """Make sure a single environment map (e.g. 'b.exr' or 'b') is in envmaps/."""
    name = name if name.endswith('.exr') else name + '.exr'
    if name not in ENVMAPS:
        raise KeyError('unknown envmap {!r}, available: {}'.format(name, sorted(ENVMAPS)))
    return _fetch(BLENDER_REPO, 'dataset', ENVMAPS[name], ENVMAP_DIR / name, force=force)


def ensure_envmaps(names=None, force=False):
    """Make sure the environment maps used by envmaps/fit_envmap_with_sg.py are present."""
    return [ensure_envmap(name, force=force) for name in (names or ENVMAPS)]


def download_blender_data(local_dir, force=False):
    """Download the raw Blender scenes (.blend + textures + all HDRIs) for re-rendering."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            'huggingface_hub is required to download the assets. '
            'Install it with "pip install -U huggingface_hub".'
        )
    print('downloading {} -> {} (~770 MB)'.format(BLENDER_REPO, local_dir))
    return snapshot_download(
        repo_id=BLENDER_REPO,
        repo_type='dataset',
        local_dir=str(local_dir),
        force_download=force,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--weights', action='store_true',
                        help='download the SuperPoint/SuperGlue checkpoints (~92 MB)')
    parser.add_argument('--envmaps', action='store_true',
                        help='download the b/c/d HDRI environment maps (~57 MB)')
    parser.add_argument('--blender-data', metavar='DIR', nargs='?', const='blender_data',
                        help='also download the raw Blender scenes to DIR (~770 MB, '
                             'default: ./blender_data)')
    parser.add_argument('--force', action='store_true',
                        help='re-download even if the file already exists')
    args = parser.parse_args(argv)

    # no group selected -> fetch everything the training/preprocessing code needs
    if not (args.weights or args.envmaps or args.blender_data):
        args.weights = args.envmaps = True

    if args.weights:
        ensure_keypoint_weights(force=args.force)
    if args.envmaps:
        ensure_envmaps(force=args.force)
    if args.blender_data:
        download_blender_data(args.blender_data, force=args.force)

    print('done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
