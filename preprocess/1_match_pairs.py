import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import glob
import itertools
import argparse 
import shutil
import cv2 
import torch

from tqdm import tqdm
from pathlib import Path
import numpy as np
import imageio.v2 as imageio 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from termcolor import colored

from utils.pose_transform import *
from keypoint_matching.matching import Matching
from keypoint_matching.utils import (make_matching_plot, AverageTimer,  read_and_rotate_image )

def t2n(t):
    return t.detach().cpu().numpy()

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
        '--is_synthetic', type=int, default=0, help='whether the object is synthetic')
    parser.add_argument(
        '--train_res', type=int, default=800, help='training resolution for nerf')
    parser.add_argument(
        '--instance_num', type=int, default=7, required=True, help='number of instances in the image')
    parser.add_argument(
        '--rotate_delta_angle', type=int, default=6, help='rotation delta angle for matching pairs')
    parser.add_argument(
        '--instance_dir', type=str, default=None, help='instance directory')
    
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

    cur_path = sys.path[0]

    ############################ Input ############################   

    # select_name = opt.select_name
    # assert select_name is not None
    # print(colored('Select object: {}'.format(select_name), 'green', attrs=['bold']))
    delta_angle = opt.rotate_delta_angle

    do_match = True
    do_viz = True
    save_matches = True

    is_synthetic = opt.is_synthetic
    real_world = not is_synthetic
    train_resolution = opt.train_res
    same_obj_num = opt.instance_num
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

    # create output directory
    output_path = os.path.join(temp_dir, 'pairwise_match') # save the matches in .npz format
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write pairwise matches to directory \"{}\"'.format(output_dir))

    # create plot directory
    match_plot_path = os.path.join(temp_dir, 'pairwise_match_viz') 
    if os.path.exists(match_plot_path) and do_match and do_viz:
        shutil.rmtree(match_plot_path) 
    match_plot_dir = Path(match_plot_path)
    match_plot_dir.mkdir(exist_ok=True, parents=True)
    print('Will write visualization to directory \"{}\"'.format(match_plot_dir))

    center_dir = os.path.join(temp_dir, '*.npy')
    center_path = glob.glob(center_dir)
    assert len(center_path) == 1, 'center_path should be unique'
    center_path = center_path[0]
    crop_size = int(os.path.basename(center_path).split('_')[-1].split('.')[0])
    index2center = np.load(center_path, allow_pickle=True).item()
    assert (same_obj_num-1) in index2center.keys(), 'the number of instances should be the same as the instance number in the image'

    half_crop_size = (crop_size - 1) / 2.0

    images_paths = glob.glob(os.path.join(temp_dir, 'cropped', '*'))
    assert len(images_paths) == same_obj_num, 'number of cropped images should be the same as the instance number'
    images_indexs = list(np.arange(len(images_paths)))
    num_pairs = len(images_paths) * (len(images_paths) - 1) // 2

    timer = AverageTimer(newline=True) 
    iteration_object = itertools.combinations(images_indexs, 2)
    for pair in tqdm(iteration_object, total=num_pairs):
        print('Match pair {} and {}'.format(pair[0], pair[1])) 
        
        name0 = images_paths[pair[0]]
        name1 = images_paths[pair[1]] 
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1) 

        center0 = np.array(index2center[pair[0]])[[1, 0]] # reverse to x (width), y (height)
        center1 = np.array(index2center[pair[1]])[[1, 0]] # reverse to x (width), y (height) 

        if not (do_match or do_viz):
            timer.print('Finished pair {:5} of {:5}'.format(str(pair), num_pairs))
            continue 

        out_matches = {}
        # Load the image pair. 
        image0_origin = read_and_rotate_image(name0, device,  rot_angle=0)[0]
        image0, inp0 = read_and_rotate_image(name0, device, rot_angle=0)
        image1_origin = read_and_rotate_image(name1, device, rot_angle=0)[0]
        # loop over all the rotation
        R = None
        for rot in tqdm(range(0, 360, delta_angle)):

            image1, inp1 = read_and_rotate_image(name1, device, rot_angle=rot) # only rotate image1 to create relative rotation

            image0_resize = cv2.resize(image0_origin, (h, w))
            image1_resize = cv2.resize(image1_origin, (h, w))

            if image0 is None or image1 is None:
                print('Problem reading image pair: {} {}'.format(name0, name1))
                exit(1)
            timer.update('load_image')

            inv_rotation_matrix1 = cv2.getRotationMatrix2D((crop_size /2, crop_size /2), -rot, 1.0)

            if do_match:
                with torch.no_grad():
                    # Perform the matching.
                    pred = matching({'image0': inp0, 'image1': inp1})
                    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
                    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                    matches, conf = pred['matches0'], pred['matching_scores0']

                    timer.update('matcher')

                # Keep the matching keypoints.
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]

                # visualize the matches with text description
                center0 = np.array(index2center[pair[0]])[[1, 0]] # reverse to x (width), y (height)
                center1 = np.array(index2center[pair[1]])[[1, 0]] # reverse to x (width), y (height)

                ones = np.ones(shape=(len(mkpts1), 1)) 
                mkpts0_rot = mkpts0
                mkpts1_rot = np.hstack([mkpts1, ones])
                mkpts1_rot = inv_rotation_matrix1.dot(mkpts1_rot.T).T
                mkpts0back = (mkpts0_rot - half_crop_size + center0)  / downsample_rate
                mkpts1back = (mkpts1_rot - half_crop_size + center1) / downsample_rate 

                # compute the essential matrix to get the relative pose
                try:
                    M_intrinsic = np.array([[1000, 0, h//2], 
                                            [0, 1000, w//2],
                                            [0, 0, 1]]) # placeholder
                    predE, cur_inlier = cv2.findEssentialMat(mkpts0back, mkpts1back, M_intrinsic, cv2.RANSAC, 0.999, 0.05)  
                except:
                    cur_inlier = np.array([0])
                else:
                    if predE is not None and predE.shape[0] == 3 and predE.shape[1] == 3:
                        # print('inlier: ', cur_inlier.sum(), len(cur_inlier))
                        _, R, t, _ = cv2.recoverPose(predE, mkpts0back, mkpts1back, M_intrinsic) # left hand, so transpose E

            if do_viz and pair == (0, 1):
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
                    'Rotation: {:.2f}'.format(rot)
                ]

                # the object is fixed, and match points are rotated back
                viz_path = match_plot_dir / 'fixed_{}_{}_matches_{}.{}'.format(stem0, stem1, rot, 'png')
                make_matching_plot(
                    image0, image1,
                    kpts0, kpts1, mkpts0_rot, mkpts1_rot, color,
                    text, viz_path, opt.show_keypoints,
                    opt.fast_viz, opt.opencv_display, 'Matches', small_text)
                
                # show the rotation process
                viz_path = match_plot_dir / 'rotate_{}_{}_matches_{}.{}'.format(stem0, stem1, rot, 'png')
                make_matching_plot(
                    (inp0.detach().cpu().numpy()[0,0] * 255).astype(np.uint8), (inp1.detach().cpu().numpy()[0,0] * 255).astype(np.uint8),
                    kpts0, kpts1, mkpts0, mkpts1, color,
                    text, viz_path, opt.show_keypoints,
                    opt.fast_viz, opt.opencv_display, 'Matches', small_text)


                # visualize the matches without text description
                # the image is rotated, so the keypoints should be rotated back
                # clean plot
                # small_text=[]
                # make_matching_plot(
                #     (inp0[0,0].detach().cpu()*255).numpy(), (inp1[0,0].detach().cpu()*255).numpy(), None, None, None, None, None,
                #     None, viz_path, opt.show_keypoints,
                #     opt.fast_viz, opt.opencv_display, 'Matches', small_text)

                timer.update('viz_match') 

            if save_matches:
                out_matches.update({str(rot):{
                    # 'keypoints0': kpts0,
                    # 'keypoints1': kpts1, # [N, 2]
                    # 'keypoints0_back': kpts0_rot, # keypoints
                    # 'keypoints0_back': kpts1_rot,
                    # 'scores0':pred['scores0'], # scores
                    # 'scores1':pred['scores1'] ,
                    # 'descriptors0':pred['descriptors0'], # [dim, N] descriptors
                    # 'descriptors1':pred['descriptors1'], 
                    'mkpts0back': mkpts0back, # (800, 800) 
                    # 'mkpts1back': mkpts1back,
                    # 'matches': matches,      # matches0
                    # 'match_confidence': conf, # matching_scores0 
                    'R': R,
                    # 't': t
                }}) 
                np.savez(str(matches_path), **out_matches)
    
    print('Done stage 1: match pairs. Please check the visualization in the directory \"{}\"'.format(match_plot_dir))