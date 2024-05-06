
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import cv2
import glob
import argparse

from tqdm import tqdm
from PIL import Image
import numpy as np
from scipy.optimize import minimize

# Define the objective function
def objective(x, range_matrix, valid_mask, weight, return_inside=False): 
    x = np.concatenate([np.array([0.0]), x])
    pred_matrix = (x.reshape(1,-1) - x.reshape(-1,1)) 
    diff_matrix0 = pred_matrix - range_matrix[..., 0]
    diff_matrix1 = pred_matrix - range_matrix[..., 1]
    diff_matrix2 = diff_matrix0 + np.pi * 2 # np.pi * 2
    diff_matrix3 = diff_matrix1 + np.pi * 2
    inside_mask0 = np.logical_xor(diff_matrix0 > 0, diff_matrix1 > 0) # when inside, xor is true
    inside_mask1 = np.logical_xor(diff_matrix2 > 0, diff_matrix3 > 0) # when inside, xor is true
    inside_mask = inside_mask0 | inside_mask1 
    loss = np.minimum(np.minimum(abs(diff_matrix0), abs(diff_matrix1)),
                      np.minimum(abs(diff_matrix2), abs(diff_matrix3)))

    # loss_center = np.minimum(abs((diff_matrix0 + diff_matrix1) / 2.0), 
    #                          abs((diff_matrix2 + diff_matrix3) / 2.0))
    
    loss[inside_mask] = 0.0 
    loss[~valid_mask] = 0.0 
    if return_inside:
        return loss * weight, inside_mask | ~valid_mask
    return np.sum(loss * weight)
    # if return_inside:
    #     return loss  , inside_mask | ~valid_mask
    # return np.sum(loss  )

def read_image(path,  rot_angle):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None 
    w, h = image.shape[1], image.shape[0]  
    rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), rot_angle, 1.0)
    rotated_img = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_img 

# Define the optimization function
def optimize_angles(range_matrix, valid_mask, weight):
    # Determine the number of vectors
    n = len(range_matrix) 
    # Set the initial guess for the relative angles
    # x0 =  * 2 * np.pi
    x0 = (pair_range_matrix[0,1:,0]).copy() % (2*np.pi) + np.random.randn(n-1) * np.pi / 2
    # Define the bounds for the relative angles
    bounds = [(0, 2 * np.pi)] * (n-1)  # 720
    # Minimize the objective function subject to the constraints
    res = minimize(objective, x0, args=(range_matrix, valid_mask, weight,), bounds=bounds, method='Powell' ) # Powell 4 
    loss, mask = objective(res.x, range_matrix, valid_mask, weight, return_inside=True,)
    # Extract the optimized relative angles
    abs_angles = np.concatenate([np.array([0.0]), res.x]) / np.pi * 180
    return loss, mask, abs_angles

if __name__ == "__main__":

    parser = argparse.ArgumentParser()   
    parser.add_argument('--instance_dir', type=str, default=None)
    parser.add_argument('--instance_num', type=int, default=None)
    parser.add_argument('--iters', type=int, default=50)
    args = parser.parse_args()

    iters = args.iters # total iterations for global rotation optimization
    instance_dir = args.instance_dir
    instance_num = int(args.instance_num)
    raw_dir = os.path.join(instance_dir, 'raw')
    temp_dir = os.path.join(raw_dir, 'temp')
    cropped_dir = os.path.join(temp_dir, 'cropped')
    
    good_pair_dir = os.path.join(temp_dir, 'good_rotation_range')
    rot_save_dir = good_pair_dir
    images_paths = glob.glob(os.path.join(cropped_dir, '*'))
    obj_num = instance_num
    assert len(images_paths) == obj_num, 'The number of images should be {}, but only {} images found.'.format(obj_num, len(images_paths))
    
    good_pair_path = os.path.join(good_pair_dir, 'good_pair_range.txt')

    pair_range_matrix = np.zeros([obj_num, obj_num, 3]) # [i,j] means j relative to i

    f_pair_range = open(good_pair_path, 'r')

    lines = f_pair_range.readlines()
    for line in lines:
        info = line.replace(':','_').replace(' ','_').split('_')
        rot0 = (float(info[4])) / 180 * np.pi  # / 180 * np.pi
        rot1 = (float(info[6])) / 180 * np.pi
        weight = float(info[-1][6:]) 
        pair_range_matrix[int(info[1]), int(info[2])] = np.array([rot0, rot1, weight]) 

        img_name0 = '{}_rgb.png'.format(str(int(info[1])).zfill(3))
        img_name1 = '{}_rgb.png'.format(str(int(info[2])).zfill(3))
    
    f_pair_range.close()

    valid_mask = pair_range_matrix[..., 0] > 0
    weight = pair_range_matrix[..., 2]
    weight = weight / weight.sum()

    best_loss = np.inf
    best_abs_angles = None
    for i in tqdm(range(iters)):
        loss, mask, abs_angles = optimize_angles(pair_range_matrix, valid_mask, weight)
        loss_sum = np.sum(loss)
        if loss_sum < best_loss:
            best_loss = loss_sum
            best_abs_angles = abs_angles 
    
    print('best loss = {}\n', (loss * 10000).astype(int))
    print('best mask = \n', mask)
    print('best angles = \n', best_abs_angles)
    with open('{}/abs_rotate.txt'.format(rot_save_dir), 'w+') as f:
        for rot in best_abs_angles:
            f.write('{}\n'.format(rot))
    
    # save the rotated images after global rotation correction
    for i in range(obj_num):
        rot_image = read_image(images_paths[i], rot_angle=best_abs_angles[i])
        img = Image.fromarray(rot_image)
        img.save(os.path.join(rot_save_dir, 'rot_{}.png'.format(i)))
    
    print('Done stage 3: optimize global rotation.')