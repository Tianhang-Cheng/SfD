
import os
import sys

# os.environ['OMP_NUM_THREADS']="1"
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import glob
import argparse

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from termcolor import colored

from utils.pose_transform import *
from datasets.data_info import obj_info
 
def longest_consecutive_sequence(nums):
    if len(nums) == 0:
        return 0, None, None

    max_len = 1
    current_len = 1
    start = 0
    end = 0

    for i in range(1, len(nums)):
        if nums[i] == nums[i-1] + 1:
            current_len += 1
            if current_len > max_len:
                max_len = current_len
                end = i
                start = end - max_len + 1
        elif nums[i] != nums[i-1]:
            if current_len > max_len:
                max_len = current_len
                end = i-1
                start = end - max_len + 1
            current_len = 1

    if current_len > max_len:
        max_len = current_len
        end = len(nums)-1
        start = end - max_len + 1

    return max_len, start, end

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()   
    parser.add_argument('--instance_dir', type=str, default=None)
    parser.add_argument('--instance_num', type=int, default=None)
    args = parser.parse_args()

    instance_dir = args.instance_dir
    n_obj = args.instance_num

    is_synthetic = False
    real_world = not is_synthetic
    raw_dir = os.path.join(instance_dir, 'raw')
    temp_dir = os.path.join(raw_dir, 'temp')

    # load data from path
    range_save_dir = os.path.join(temp_dir, 'good_rotation_range')
    if not os.path.exists(range_save_dir):
        os.makedirs(range_save_dir)
    
    pairwise_match_dir = os.path.join(temp_dir, 'pairwise_match')
    paths = glob.glob(os.path.join(pairwise_match_dir, '*_matches.npz'))
    x = len(paths) 
    assert len(paths) == n_obj * (n_obj - 1) // 2, '{} objects should have {} pairs, but only {} pairs found.'.format(n_obj, n_obj * (n_obj - 1) // 2, len(paths))
    print(colored('Filter good pairs from {} objects ({} pairs).'.format(n_obj, len(paths)), 'magenta', attrs=['bold']))
    
    # output
    f_txt_path = os.path.join(range_save_dir, 'good_pair_range.txt')
    f_txt = open(f_txt_path, 'w+')

    # read the length of matching keypoints to determine the threshold of good pairs
    mean_nums = []
    for path in paths:
        pair0 = int(os.path.basename(path)[0:3])
        pair1 = int(os.path.basename(path)[8:11])
        pair = (pair0, pair1)
        f = dict(np.load(path, allow_pickle=True))

        mkpts_num = [v.item()['mkpts0back'].shape[0] for k, v in f.items()]
        mkpts_num = np.mean(mkpts_num)
        mean_nums.append(mkpts_num)
    
    mean_nums = np.array(mean_nums)

    print('Keypoints number distribution of each pair: min = {:.1f}, max = {:3.1f}, median = {:3.1f}, mean = {:3.1f}, std = {:3.1f}'.format(
        np.min(mean_nums), np.max(mean_nums), np.median(mean_nums), np.mean(mean_nums), np.std(mean_nums)))

    # check each pair
    for path in paths:
        pair0 = int(os.path.basename(path)[0:3])
        pair1 = int(os.path.basename(path)[8:11])
        pair = (pair0, pair1)
 
        # ['keypoints0', 'keypoints1', 'keypoints0_back',
        # 'scores0', 'scores1', 'descriptors0', 'descriptors1',
        # 'mkpts0back', 'mkpts1back', 'matches', 'match_confidence']
        f = dict(np.load(path, allow_pickle=True))
        mkpts_num = [v.item()['mkpts0back'].shape[0] for k, v in f.items()]
        mkp_threshold = np.median(mkpts_num) * 1.2 # if the number of keypoints is less than this threshold, then this rotation state is not good.

        drot = 360 // len(f)

        valid = False

        good_rot = []
        good_rot_kpnum = []

        for rot in tqdm(range(0, 360, drot)):
            mkpt_lens = len(f[str(rot)].item()['mkpts0back'])

            R = f[str(rot)].item()['R']
            if R is None:
                continue  

            if mkpt_lens < mkp_threshold:
                continue

            good_rot.append(rot // drot) 
            good_rot_kpnum.append(mkpt_lens)

            valid = True   
        
        # max consequence  
        good_rot = np.array(good_rot)
        good_rot = np.concatenate([good_rot, good_rot+360//drot])
        _, start, end = longest_consecutive_sequence(good_rot) 

        good_rot_kpnum = np.array(good_rot_kpnum) 
        good_rot_kpnum = np.concatenate([good_rot_kpnum, good_rot_kpnum])
        
        if valid: 
            avg_kpnum = np.mean(good_rot_kpnum[start:end+1])
            
            r_start = good_rot[start] * drot
            r_end = good_rot[end] * drot

            if abs(r_end - r_start) == 0:
                # the rotation range is smaller than expected 
                s = r_start - drot
                e = r_end + drot
                text = 'pair {}_{}_rot_{}_to_{}: avg_kpnum={}\n'.format(pair0, pair1, s, e, avg_kpnum)
            else:
                # the rotation range is larger than expected, just use the range
                s = r_start
                e = r_end
                text = 'pair {}_{}_rot_{}_to_{}: avg_kpnum={}\n'.format(pair0, pair1, s, e, avg_kpnum)

            print('pair {}_{}: threshold = {:.2f}, range = ({}, {})'.format(pair0, pair1, mkp_threshold, s, e))
            f_txt.write(text)
        else:
            print('pair {}_{}: threshold = {:.2f}, no meaningful correspondence.'.format(pair0, pair1, mkp_threshold)) 

    f_txt.close()
    print('Done stage 2. Filter pairs.')