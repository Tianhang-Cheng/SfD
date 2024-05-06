import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import glob
import itertools
import argparse 
import torch
import cv2 

from tqdm import tqdm
from pathlib import Path
import numpy as np
import imageio.v2 as imageio 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from termcolor import colored

from utils.pose_transform import *
from keypoint_matching.matching import Matching
from keypoint_matching.utils import (make_matching_plot, AverageTimer,   read_and_rotate_image )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize') 

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor', 
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.003, # 0.005
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.1, # 0.2
        help='SuperGlue match threshold') 
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches') 
    parser.add_argument(
        '--fast_viz', default='true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--select_name', type=str, default=None, help='object name')
    parser.add_argument(
        '--absolute_mkp_threshold', type=int, default=8, help='the threshold of the number of keypoints')
    parser.add_argument(
        '--train_res', type=int, default=800, help='training resolution for nerf')
    parser.add_argument(
        '--instance_num', type=int, default=7, required=True, help='number of instances in the image')
    parser.add_argument(
        '--instance_dir', type=str, default=None, help='instance directory'
    )
    opt = parser.parse_args()
    # print(opt)

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    ############################ Input ############################   

    absolute_mkp_threshold = opt.absolute_mkp_threshold

    do_match = True
    do_viz = True
    save_matches = True

        
    # read the object information
    train_resolution = opt.train_res
    same_obj_num = opt.instance_num
    is_synthetic = False
    real_world = not is_synthetic

    instance_dir = opt.instance_dir
    raw_dir = os.path.join(instance_dir, 'raw')
    temp_dir = os.path.join(raw_dir, 'temp')

    # read the image size and calculate the downsample rate
    raw_image_path = os.path.join(temp_dir, '000_rgb.png')
    h, w, _ = imageio.imread(raw_image_path).shape # raw image size
    assert h == w, 'image should be square'
    downsample_rate = h / train_resolution
    assert downsample_rate >= 1, 'downsample_rate should be >= 1'
    print(colored('Raw image shape is ({}, {}), training resolution is ({}, {}), with the downsample rate {}'.format(h, w, train_resolution, train_resolution, downsample_rate), 'magenta', attrs=['bold']))

    # create the output directory
    output_path = os.path.join(temp_dir, 'sfm_inputs')
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    matches_path = output_dir / 'final_matches.npz'
    feats_path = output_dir / 'final_feats.npz'
    pairs_path = output_dir / 'image_pair.txt'
    f_pair_name = open(pairs_path, 'w+') # w+: create a new file if it does not exist in path, will overwrite the file if it exists
    print('Will write matches to directory \"{}\"'.format(output_dir))

    # create plot directory
    match_plot_path = os.path.join(temp_dir, 'global_match_viz')
    match_plot_dir = Path(match_plot_path)
    match_plot_dir.mkdir(exist_ok=True, parents=True)
    print('Will write visualization to directory \"{}\"'.format(match_plot_dir))

    # read the good pairs and global rotation
    good_pairs = []
    abs_rotates = []
    good_pair_dir = os.path.join(temp_dir, 'good_rotation_range')
    good_pair_path = os.path.join(good_pair_dir, 'good_pair_range.txt')
    abs_rotate_path = os.path.join(good_pair_dir, 'abs_rotate.txt')
    if os.path.exists(good_pair_path): 
        with open(good_pair_path, 'r') as f_:
            lines = f_.readlines()
            for line in lines:
                line = line.split(' ')[1].split('_')
                good_pairs.append([int(line[0]), int(line[1])])
    else:
        raise ValueError('good_pair_path does not exist')
    if os.path.exists(abs_rotate_path):
        with open(abs_rotate_path, 'r') as f_:
            lines = f_.readlines()
            for line in lines:
                abs_rotates.append(float(line.replace('\n','')))       
        abs_rotates = np.array(abs_rotates)
    else:
        raise ValueError('abs_rotate_path does not exist')

    center_path = glob.glob(os.path.join(temp_dir, '*.npy'))
    assert len(center_path) == 1, 'center_path should be unique'
    center_path = center_path[0]
    crop_size = int(os.path.basename(center_path).split('_')[-1].split('.')[0])
    index2center = np.load(center_path, allow_pickle=True).item()
    assert (same_obj_num-1) in index2center.keys()
    half_crop_size = (crop_size - 1) / 2.0

    crop_dir = os.path.join(temp_dir, 'cropped')
    images_paths = glob.glob(os.path.join(crop_dir, '*.png'))
    images_indexs = list(np.arange(len(images_paths)))
    num_pairs = len(images_paths) * (len(images_paths) - 1) // 2

    timer = AverageTimer(newline=True)

    out_matches = {}
    out_feats = {}
    deletes = []

    for pair in good_pairs: 
        print('Match pair {} and {}'.format(pair[0], pair[1])) 
        
        name0 = images_paths[pair[0]]
        name1 = images_paths[pair[1]] 
        stem0, stem1 = Path(name0).stem, Path(name1).stem 

        if not (do_match or do_viz):
            timer.print('Finished pair {:5} of {:5}'.format(str(pair), num_pairs))
            continue

        # Load the image pair.  
        image0_origin, inp0 = read_and_rotate_image(name0, device, rot_angle=abs_rotates[pair[0]]) 
        image1_origin, inp1 = read_and_rotate_image(name1, device, rot_angle=abs_rotates[pair[1]])  

        image0_resize = cv2.resize(image0_origin, (h, w))
        image1_resize = cv2.resize(image1_origin, (h, w))

        if image0_origin is None or image1_origin is None:
            print('Problem reading image pair: {} {}'.format(name0, name1))
            exit(1)
        timer.update('load_image')

        inv_rotation_matrix0 = cv2.getRotationMatrix2D((crop_size /2, crop_size /2), -abs_rotates[pair[0]], 1.0)
        inv_rotation_matrix1 = cv2.getRotationMatrix2D((crop_size /2, crop_size /2), -abs_rotates[pair[1]], 1.0)

        if do_match:
            with torch.no_grad():
                # Perform the matching.
                pred = matching({'image0': inp0, 'image1': inp1})
                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
                kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                matches0, conf0 = pred['matches0'], pred['matching_scores0']
                matches1, conf1 = pred['matches1'], pred['matching_scores1']
                timer.update('matcher')

        # Keep the matching keypoints.
        valid = matches0 > -1 
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches0[valid]]
        mconf = conf0[valid]

        if len(mkpts0) < 8:
            deletes.append(pair)
            continue

        f_pair_name.write('{}.png {}.png\n'.format(stem0, stem1))

        center0 = np.array(index2center[pair[0]])[[1, 0]] # reverse to x (width), y (height)
        center1 = np.array(index2center[pair[1]])[[1, 0]] # reverse to x (width), y (height)

        ones = np.ones(shape=(len(kpts0), 1)) 
        kpts0_rot = np.hstack([kpts0, ones])
        kpts0_rot = inv_rotation_matrix0.dot(kpts0_rot.T).T
        kpts0back = (kpts0_rot - half_crop_size + center0)  / downsample_rate 

        ones = np.ones(shape=(len(mkpts0), 1)) 
        mkpts0_rot = np.hstack([mkpts0, ones])
        mkpts0_rot = inv_rotation_matrix0.dot(mkpts0_rot.T).T
        mkpts0back = (mkpts0_rot - half_crop_size + center0)  / downsample_rate 

        ones = np.ones(shape=(len(kpts1), 1)) 
        kpts1_rot = np.hstack([kpts1, ones])
        kpts1_rot = inv_rotation_matrix1.dot(kpts1_rot.T).T
        kpts1back = (kpts1_rot - half_crop_size + center1) / downsample_rate 

        ones = np.ones(shape=(len(mkpts1), 1)) 
        mkpts1_rot = np.hstack([mkpts1, ones])
        mkpts1_rot = inv_rotation_matrix1.dot(mkpts1_rot.T).T
        mkpts1back = (mkpts1_rot - half_crop_size + center1) / downsample_rate 
        
        # update the output dict (matches and features)
        if str(pair[0]) not in out_feats.keys(): 
            out_feats.update({str(pair[0]):{
                'keypoints': kpts0, # [N, 2]
                'keypoints_back': kpts0back, # keypoints
                'scores':pred['scores0'], # scores
                'descriptors':pred['descriptors0'], # [dim, N] descriptors
            }}) 
        if str(pair[1]) not in out_feats.keys(): 
            out_feats.update({str(pair[1]):{
                'keypoints': kpts1,
                'keypoints_back': kpts1back, # keypoints
                'scores':pred['scores1'], # scores
                'descriptors':pred['descriptors1'], # [dim, N] descriptors
            }}) 

        out_matches.update({str('{}_{}'.format(pair[0], pair[1])):{    
            'mkpts0back': mkpts0back, # (800, 800) 
            'mkpts1back': mkpts1back,
            'matches0': matches0,      # matches0
            'match_confidence0': conf0, # matching_scores0
        }})

        if do_viz:
            viz_path = match_plot_dir / '{}_{}_matches.{}'.format(stem0, stem1, 'png')
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
                'Rotation: {:.2f}-{:.2f}'.format(pair[0], pair[1])
            ]
            make_matching_plot(
                image0_origin, image1_origin, kpts0, kpts1, mkpts0_rot, mkpts1_rot, color,
                text, viz_path, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches', small_text)

            # visualize the matches without text description
            # the image is rotated, so the keypoints should be rotated back
            # small_text=[]
            # make_matching_plot(
            #     (inp0[0,0].detach().cpu()*255).numpy(), (inp1[0,0].detach().cpu()*255).numpy(), None, None, None, None, None,
            #     None, viz_path, opt.show_keypoints,
            #     opt.fast_viz, opt.opencv_display, 'Matches', small_text)

            timer.update('viz_match')

    if save_matches:

        np.savez(str(matches_path), **out_matches) 
        np.savez(str(feats_path), **out_feats) 
        print('These edges has less than 8 matches: ', deletes)

    f_pair_name.close()
    print('Done stage 4: match pairs final.')