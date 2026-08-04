import os
import sys
sys.path.append('../SfD') 

import argparse
import random
import torch

import imageio.v2 as imageio
import numpy as np

from trainer.train_geometry import GeometryTrainRunner
from trainer.train_visibility import VisbilityTrainRunner
from trainer.train_material import MaterialTrainRunner
from configs.config import Config, recursive_update_strict, parse_cmdline_arguments

from datasets.data_info import obj_info

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='')
    parser.add_argument('--exps_folder_name', type=str, default='exps')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--trainstage', type=str, default='') 
    parser.add_argument('--eval', default=False, action="store_true")
    parser.add_argument('--eval_relight', default=False, action="store_true") 
    parser.add_argument('--envmap_name', type=str, default='') 
    parser.add_argument('--single_image', default=False, action="store_true") 
    parser.add_argument('--to_mesh', default=False, action="store_true")
    parser.add_argument('--to_uv', default=False, action="store_true")
    parser.add_argument('--use_pretrain_normal', default=False, action="store_true") 

    parser.add_argument('--data_split_dir', type=str, default='')
    parser.add_argument('--frame_skip', type=int, default=1, help='skip frame when training')
    parser.add_argument('--eval_frame_skip', type=int, default=1, help='skip frame when evaluation')
    parser.add_argument('--forbid_vis',default=False, action="store_true", help='PhySG')
    parser.add_argument('--init_method', type=str, help="['GT', 'GT_with_noise', 'SFM', 'Random']")
    
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--visible_num', type=int, default=-1, help='number of visible objects')
    parser.add_argument('--max_niter', type=int, default=300001, help='max number of iterations to train for')
    parser.add_argument('--select_index', type=int, default=-1, help='plot a certain image at given pose index')
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint epoch number of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--debug', default=False, action="store_true" ) 
    parser.add_argument('--train_pose', default=False, action="store_true")
    parser.add_argument('--same_obj_num', type=int, default=None,
                        help='number of duplicate instances in the image. Defaults to the entry '
                             'in datasets/data_info.py for --expname, and for an unknown expname '
                             'to the instance count read from train/000_instance_seg.png')
    parser.add_argument('--real_world', default=None, action=argparse.BooleanOptionalAction,
                        help='captured photo (loads train/000_rgb.png) instead of a synthetic '
                             'render (loads train/000_rgb.exr). Defaults to the entry in '
                             'datasets/data_info.py for --expname, and for an unknown expname '
                             'to whether train/000_rgb.exr exists')
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


def resolve_dataset_info(args):
    """
    Work out (same_obj_num, real_world) for this run.

    Explicit command line flags win, then datasets/data_info.py, then the data on disk. The
    fallback is what makes it possible to train on your own preprocessed object without first
    adding it to the obj_info table.
    """
    same_obj_num, real_world = args.same_obj_num, args.real_world

    info = obj_info.get(args.expname)
    if info is not None:
        same_obj_num = info[0] if same_obj_num is None else same_obj_num
        real_world = (not info[1]) if real_world is None else real_world

    if same_obj_num is None:
        seg_path = os.path.join(args.data_split_dir, 'train', '000_instance_seg.png')
        if not os.path.exists(seg_path):
            raise SystemExit(
                'cannot determine the number of instances: {!r} is not in obj_info '
                '(datasets/data_info.py) and {} does not exist. Pass --same_obj_num N.'.format(
                    args.expname, seg_path))
        same_obj_num = len(np.unique(imageio.imread(seg_path))) - 1  # label 0 is background
        print('read same_obj_num = {} from {}'.format(same_obj_num, seg_path))

    if real_world is None:
        real_world = not os.path.exists(
            os.path.join(args.data_split_dir, 'train', '000_rgb.exr'))
        print('assuming real_world = {} (train/000_rgb.exr {})'.format(
            real_world, 'missing' if real_world else 'found'))

    return same_obj_num, real_world

if __name__ == '__main__':

    seed_torch()
    
    args, cfg_cmd = parse_args()
    cfg = Config(args.conf)
    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)
    cfg.path = args.conf # save the config file path

    same_obj_num, real_world = resolve_dataset_info(args)

    render_dict = {
        'Geo': GeometryTrainRunner,
        'Vis': VisbilityTrainRunner,
        'Mat': MaterialTrainRunner,
    }

    trainrunner = render_dict[args.trainstage](
        conf=cfg,
        exps_folder_name=args.exps_folder_name,
        expname=args.expname,
        data_split_dir=args.data_split_dir, 
        frame_skip=args.frame_skip,
        eval_frame_skip=args.eval_frame_skip,
        batch_size=args.batch_size,
        max_niters=args.max_niter,
        same_obj_num=same_obj_num,
        visible_num=args.visible_num,
        is_continue=args.is_continue,
        timestamp=args.timestamp,
        checkpoint=args.checkpoint, 
        select_index=args.select_index,
        is_eval=args.eval,
        is_eval_relight=args.eval_relight, 
        forbid_vis=args.forbid_vis,
        init_method=args.init_method,
        single_image=args.single_image,
        real_world=real_world,
        debug=args.debug,
        train_pose=args.train_pose,
        use_pretrain_normal=args.use_pretrain_normal,
        to_uv=args.to_uv,
        to_mesh=args.to_mesh,
    )
    
    if args.eval:
        trainrunner.evaluate_envmap()
        trainrunner.evaluate()
    elif args.eval_relight:
        trainrunner.evaluate_relight('b')
        trainrunner.evaluate_relight('d')
    elif args.to_mesh:
        raise NotImplementedError
    elif args.to_uv: 
        raise NotImplementedError
    else:
        trainrunner.run()