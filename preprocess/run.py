"""Run the whole preprocessing pipeline for one object with a single command.

    python preprocess/run.py --instance_dir data/your_object

Everything is optional: the input image (raw/000_rgb.exr or raw/000_rgb.png) and the
number of instances are detected from the data on disk. Override any of it explicitly:

    python preprocess/run.py --instance_dir data/my_pile --instance_num 7 \
        --crop_size 1000 --train_res 800 --rotate_delta_angle 4

Individual stages can be re-run without redoing the expensive matching, e.g. to iterate
on the SfM step only:

    python preprocess/run.py --instance_dir data/my_pile --stages 5-7

Stages:
    0  mask and crop each instance
    1  pairwise SuperPoint + SuperGlue matching over rotations   (needs the SuperGlue weights)
    2  keep the good rotation ranges
    3  optimize the global rotations
    4  final matching with the optimized rotations               (needs the SuperGlue weights)
    5  COLMAP SfM
    6  extract the point cloud and non_empty_indexes.txt
    7  dump the object / camera poses
    8  Omnidata monocular normals                                (optional, needs Omnidata)
"""

import argparse
import os
import subprocess
import sys
from typing import List, Optional, Sequence

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from termcolor import colored

PREPROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PREPROCESS_DIR)

# stage index -> (script name, argv template). Stage 0 is driven by --image_path rather than
# --instance_dir, and only stage 1 takes --rotate_delta_angle, so each stage lists its own args.
COMMON = ['--instance_dir', '{instance_dir}', '--instance_num', '{instance_num}',
          '--train_res', '{train_res}']
STAGES = {
    0: ('0_mask_and_crop.py', ['--image_path', '{image_path}', '--crop_size', '{crop_size}',
                               '--instance_num', '{instance_num}', '--train_res', '{train_res}']),
    1: ('1_match_pairs.py', COMMON + ['--rotate_delta_angle', '{rotate_delta_angle}']),
    2: ('2_filter_pairs.py', COMMON),
    3: ('3_optimize_global_rotation.py', COMMON),
    4: ('4_match_pairs_final.py', COMMON),
    5: ('5_sfm.py', COMMON),
    6: ('6_extract_sfm_point_cloud.py', COMMON),
    7: ('7_extract_sfm_pose_and_visualize.py', COMMON),
    8: ('8_extract_monocular_cues.py', COMMON),
}
# stage 8 only produces an optional input (000_normal_pretrain.png), so a failure there is
# not fatal for the rest of the pipeline.
OPTIONAL_STAGES = (8,)


def parse_stages(spec: str) -> List[int]:
    """
    Turn a stage selection like '0-8', '5-7' or '0,5,6,7' into a sorted list of stage indexes.
    """
    stages = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = part.split('-', 1)
            stages.update(range(int(lo), int(hi) + 1))
        else:
            stages.add(int(part))
    unknown = sorted(stages - set(STAGES))
    if unknown:
        raise ValueError('unknown stage(s) {}, valid stages are 0-8'.format(unknown))
    return sorted(stages)


def find_image_path(instance_dir: str) -> str:
    """
    Locate the input image of an object, preferring the HDR .exr over the 8-bit .png.
    """
    raw_dir = os.path.join(instance_dir, 'raw')
    for name in ('000_rgb.exr', '000_rgb.png'):
        path = os.path.join(raw_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        'no 000_rgb.exr or 000_rgb.png in {}. Put your image and its instance segmentation '
        '(000_instance_seg.png) there first.'.format(raw_dir))


def count_instances(instance_dir: str) -> int:
    """
    Count the instances in raw/000_instance_seg.png (label 0 is the background).
    """
    seg_path = os.path.join(instance_dir, 'raw', '000_instance_seg.png')
    if not os.path.exists(seg_path):
        raise FileNotFoundError(
            'cannot infer --instance_num because {} does not exist. Pass --instance_num '
            'explicitly.'.format(seg_path))
    import imageio.v2 as imageio
    seg = imageio.imread(seg_path)
    return len(np.unique(seg)) - 1


def build_command(stage: int, subs: dict) -> List[str]:
    """
    Build the argv for one stage, substituting the {placeholders} in its argument template.
    """
    script, template = STAGES[stage]
    return [sys.executable, os.path.join(PREPROCESS_DIR, script)] + \
        [a.format(**subs) for a in template]


def run_stage(stage: int, subs: dict, dry_run: bool = False) -> int:
    """
    Run one preprocessing stage as a subprocess and return its exit code.
    """
    argv = build_command(stage, subs)
    print(colored('\n[stage {}] {}'.format(stage, ' '.join(argv)), 'cyan', attrs=['bold']))
    if dry_run:
        return 0
    # argv is a list, so shell=False is correct here. Passing a single string instead only
    # works on Windows -- on Linux/macOS the whole string is taken as the program name.
    return subprocess.run(argv, cwd=REPO_DIR).returncode


def describe_missing_instance_dir(instance_dir: str) -> str:
    """
    Explain that an object directory is missing, listing the objects that do exist next to it.

    Only data/airplane and data/your_object ship with the repository, so pointing --instance_dir
    at an object name taken from the paper or from datasets/data_info.py is a common mistake.
    """
    message = '{} is not a directory.'.format(instance_dir)
    parent = os.path.dirname(os.path.normpath(instance_dir)) or '.'
    if os.path.isdir(parent):
        found = sorted(name for name in os.listdir(parent)
                       if os.path.isdir(os.path.join(parent, name)))
        message += ' Objects in {}: {}.'.format(parent, ', '.join(found) if found else 'none')
    return message + (' Create it with raw/000_rgb.png (or raw/000_rgb.exr) and '
                      'raw/000_instance_seg.png inside, or download a pre-processed object from '
                      'the DuplicateSingleImage dataset -- see the README.')


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Parse the arguments and run the selected preprocessing stages in order.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__[__doc__.index('Stages:'):])
    parser.add_argument('--instance_dir', type=str, default=None,
                        help='object directory, e.g. data/your_object. It must contain '
                             'raw/000_rgb.{exr,png} and raw/000_instance_seg.png')
    parser.add_argument('--object_name', type=str, default=None,
                        help='object name; --instance_dir defaults to <data_folder>/<object_name>')
    parser.add_argument('--data_folder', type=str, default='data',
                        help='parent folder of the objects (default: data)')
    parser.add_argument('--image_path', type=str, default=None,
                        help='input image (default: <instance_dir>/raw/000_rgb.exr, '
                             'falling back to 000_rgb.png)')
    parser.add_argument('--instance_num', type=int, default=None,
                        help='number of instances in the image (default: read from '
                             'raw/000_instance_seg.png)')
    parser.add_argument('--crop_size', type=int, default=1200,
                        help='crop size per instance, used for feature matching. Too small and '
                             'the instance gets cut off (stage 0 checks this for you)')
    parser.add_argument('--train_res', type=int, default=800,
                        help='training resolution for the NeuS stage; may differ from --crop_size')
    parser.add_argument('--rotate_delta_angle', type=int, default=8,
                        help='rotation step for the matching pairs. Total rotations = '
                             '360 // rotate_delta_angle * instance_num ** 2, so halving this '
                             'quadruples stage 1 runtime but often recovers more instances')
    parser.add_argument('--stages', type=str, default='0-8',
                        help="stages to run, e.g. '0-8' (default), '5-7' or '0,5,6,7'")
    parser.add_argument('--dry_run', action='store_true',
                        help='print the commands without running them')
    args = parser.parse_args(argv)

    if args.instance_dir is None:
        if args.object_name is None:
            parser.error('pass --instance_dir (or --object_name)')
        args.instance_dir = os.path.join(args.data_folder, args.object_name)
    if not os.path.isdir(args.instance_dir):
        parser.error(describe_missing_instance_dir(args.instance_dir))

    stages = parse_stages(args.stages)
    image_path = args.image_path or find_image_path(args.instance_dir)
    instance_num = args.instance_num
    if instance_num is None:
        instance_num = count_instances(args.instance_dir)
        print(colored('Detected {} instances in raw/000_instance_seg.png'.format(instance_num),
                      'magenta', attrs=['bold']))

    subs = {
        'instance_dir': args.instance_dir,
        'instance_num': str(instance_num),
        'train_res': str(args.train_res),
        'crop_size': str(args.crop_size),
        'rotate_delta_angle': str(args.rotate_delta_angle),
        'image_path': image_path,
    }

    print(colored('Preprocessing {} ({} instances, image {}), stages {}'.format(
        args.instance_dir, instance_num, image_path, stages), 'green', attrs=['bold']))

    for stage in stages:
        code = run_stage(stage, subs, dry_run=args.dry_run)
        if code == 0:
            continue
        if stage in OPTIONAL_STAGES:
            print(colored('[stage {}] failed (exit {}); it is optional, continuing. Train '
                          'without --use_pretrain_normal.'.format(stage, code),
                          'yellow', attrs=['bold']))
            continue
        print(colored('[stage {}] failed with exit code {}; stopping.'.format(stage, code),
                      'red', attrs=['bold']))
        return code

    print(colored('\nPreprocessing finished! Train with:\n'
                  '  python exp_runner.py --conf configs/default.yaml --data_split_dir {} '
                  '--expname {} --trainstage Geo --init_method SFM'.format(
                      args.instance_dir, os.path.basename(os.path.normpath(args.instance_dir))),
                  'green', attrs=['bold']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
