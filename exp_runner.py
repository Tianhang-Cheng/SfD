import os
import sys
sys.path.append('../SfD') 

import argparse
import random
import torch

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
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd

if __name__ == '__main__':

    seed_torch()
    
    args, cfg_cmd = parse_args()
    cfg = Config(args.conf)
    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)
    cfg.path = args.conf # save the config file path

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
        same_obj_num=obj_info[args.expname][0],
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
        real_world=not obj_info[args.expname][1],
        debug=args.debug,
        train_pose=args.train_pose,
        use_pretrain_normal=args.use_pretrain_normal,
        to_uv=args.to_uv,
        to_mesh=args.to_mesh,
    )
    
    if args.eval:
        raise NotImplementedError
        trainrunner.evaluate_envmap()
        trainrunner.evaluate()
    elif args.eval_relight:
        raise NotImplementedError
        # trainrunner.evaluate_relight(envmap_name=args.envmap_name) 
        trainrunner.evaluate_relight('b') 
        trainrunner.evaluate_relight('d') 
    elif args.to_mesh:
        raise NotImplementedError
    elif args.to_uv: 
        raise NotImplementedError
    else:
        trainrunner.run()