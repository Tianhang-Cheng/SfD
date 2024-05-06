import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import glob
import cv2
import torch
import argparse

import numpy as np
import imageio.v2 as imageio
from torchvision.ops import masks_to_boxes
import PIL.Image as Image
from termcolor import colored
from matplotlib import pyplot as plt

from utils.rend_util import load_seg

def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)

def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2)
                    + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))
    return d

def get_crop_center(img):
    if len(img.shape) == 3:
        mask = (img[..., 0]>1)
    else:
        mask = (img>1)
    bboxes = masks_to_boxes(torch.from_numpy(mask).float().unsqueeze(0))
    bboxes = bboxes.cpu().numpy().astype(np.int64)
    assert bboxes.shape[0]==1
    
    box_center = ((bboxes[0, 1]+bboxes[0, 3])/2, (bboxes[0, 0]+bboxes[0, 2])/2)

    return box_center

def crop_image(img, crop_h, crop_w): 
    mask = (img[..., 0]>1)
    bboxes = masks_to_boxes(torch.from_numpy(mask).float().unsqueeze(0))
    bboxes = bboxes.cpu().numpy().astype(np.int64)
    assert bboxes.shape[0]==1
    crop_img = img[bboxes[0, 1]: bboxes[0, 3], bboxes[0, 0]: bboxes[0, 2]]    

    canvas = np.zeros((crop_h, crop_w, 3)).astype(np.uint8)

    h, w = crop_img.shape[:2]        
    c_h = crop_h // 2
    c_w = crop_w // 2
    hh = (h+1) // 2
    ww = (w+1) // 2

    canvas_patch_size = canvas[c_h-hh:c_h-hh+h, c_w-ww:c_w-ww+w].shape[:2]
    target_size = crop_img.shape[:2]
    if canvas_patch_size != target_size:
        raise ValueError('Please increase the crop size and run again. The object is too large to be cropped.')
    canvas[c_h-hh:c_h-hh+h, c_w-ww:c_w-ww+w] = crop_img
    crop_img = canvas
    
    return crop_img

def drawlines(img1, img2, lines, pts1, pts2):
    
    c = img1.shape[1] 

    pts1 = pts1.astype(np.int32)
    pts2 = pts2.astype(np.int32)
    
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        x0, y0 = map(int, [0, -r[2] / r[1] ])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1] ])
        
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1 , img2 

def check_invalid(p, h=800, w=800):
    """
    p [n,2]
    """
    px = p[:, 0]
    py = p[:, 1]
    return (px < 0) | (px > w) | (py < 0) | (py > h)

tonemap_img = lambda x: np.power(x, 1./2.2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()   
    parser.add_argument('--image_path', type=str, default=None, help='object name')
    parser.add_argument('--crop_size', type=int, default=800, help='crop size for each instance')
    parser.add_argument('--train_res', type=int, default=800, help='training resolution for nerf')
    parser.add_argument('--instance_num', type=int, default=1, help='object name')
    args = parser.parse_args()

    obj_name = 'object' # for center
    crop_size = args.crop_size
    train_res = args.train_res
    image_path = args.image_path
    instance_num = args.instance_num
    assert obj_name is not None

    root_dir = os.path.dirname(image_path)
    name = os.path.basename(image_path)
    out_dir = os.path.join(root_dir, 'temp')

    ################# mask and resize image #################
    os.makedirs(out_dir, exist_ok=True)
    processed_data_path = os.path.join(os.path.dirname(root_dir), 'train')
    os.makedirs(processed_data_path, exist_ok=True)
    
    # reisize to 800x800 for nerf training
    resized_dir = os.path.join(out_dir, 'resized')
    if not os.path.exists(resized_dir):
        os.makedirs(resized_dir)

    image_path = os.path.join(root_dir, name) 
    rgb = imageio.imread(image_path)
    cur_mask = load_seg(os.path.join(root_dir, '000_instance_seg.png'), input_range='0_255', output_range='0_n')
    assert cur_mask.max() == instance_num, 'The number of instances in the image does not match the input instance_num'

    h, w, _ = rgb.shape
    assert h == w, 'Currently only support square images. Please crop the image to square.'
    print(colored('Raw image shape is ({}, {}), crop size is ({}, {}), training resolution is ({}, {})'.format(h, w, crop_size, crop_size, train_res, train_res), 'magenta', attrs=['bold']))


    # put resize training data to train folder
    rgb_resized = cv2.resize(rgb, (train_res, train_res))
    img = Image.fromarray(rgb_resized)
    img.save(os.path.join(processed_data_path, '000_rgb.png'.format(obj_name)))
    seg_resized = (cv2.resize(cur_mask, (train_res, train_res), interpolation=cv2.INTER_NEAREST) / cur_mask.max() * 255).astype(np.uint8)
    img = Image.fromarray(seg_resized)
    img.save(os.path.join(processed_data_path, '000_instance_seg.png'.format(obj_name)))
    mask_resize = (seg_resized > 0) * 255
    img = Image.fromarray(mask_resize.astype(np.uint8))
    img.save(os.path.join(processed_data_path, '000_mask.png'.format(obj_name)))

    for i in range(int(np.max(cur_mask))):

        masked_img_save_path = os.path.join(out_dir, '{}_rgb.png'.format(str(i).zfill(3)))
        resized_img_save_path = os.path.join(resized_dir, '{}_rgb.png'.format(str(i).zfill(3)))
        binarymask_img_save_path = os.path.join(out_dir, '{}_mask.png'.format(str(i).zfill(3)))
        print('Save masked image at {}'.format(masked_img_save_path))
        print('Save resized image at {}'.format(resized_img_save_path))

        if 'exr' in name:
            cur_rgb = np.clip(tonemap_img(rgb), 0, 1).copy()
            cur_rgb[cur_mask != i+1] = 0
            cur_rgb = (cur_rgb * 255).astype(np.uint8)
        else:
            cur_rgb = rgb.copy()[..., 0:3]
            cur_rgb[cur_mask != i+1] = 0
            cur_rgb = cur_rgb.astype(np.uint8)
        
        img = Image.fromarray(cur_rgb)
        img.save(masked_img_save_path)
        if train_res > 0:
            img_resized = Image.fromarray(cv2.resize(cur_rgb, (train_res, train_res)))
        else:
            img_resized = Image.fromarray(cur_rgb)
        img_resized.save(resized_img_save_path)

        cur_binarymask = np.zeros_like(cur_mask)
        cur_binarymask[cur_mask == i+1] = 255
        img = Image.fromarray(cur_binarymask.astype(np.uint8))  
        img.save(binarymask_img_save_path)
    
    # ################# crop image #################
    data_dir = out_dir

    cropped_dir = os.path.join(data_dir, 'cropped')
    rgb_img_list = sorted(glob.glob(os.path.join(data_dir, '*_rgb.png')))
    max_box = 0 
    for idx, i in enumerate(rgb_img_list):    
        print('Save {} cropped image'.format(idx))
        img = imageio.imread(i)
        bname = os.path.basename(i) 

        if not os.path.exists(cropped_dir):
            os.makedirs(cropped_dir)
             
        crop_img = crop_image(img, crop_h=crop_size, crop_w=crop_size)
        cur_max = np.array(crop_img.shape[:2]).max()
        max_box = cur_max if cur_max > max_box else max_box
        imageio.imwrite(os.path.join(cropped_dir, bname), crop_img) 
    
    cropped_dir = os.path.join(data_dir, 'cropped_masks')
    rgb_img_list = sorted(glob.glob(os.path.join(data_dir, '*_mask.png')))
    for idx, i in enumerate(rgb_img_list):    
        print('Save {} cropped mask image'.format(idx))
        img = imageio.imread(i)
        bname = os.path.basename(i) 

        if not os.path.exists(cropped_dir):
            os.makedirs(cropped_dir)
             
        crop_img = crop_image(img[..., None], crop_h=crop_size, crop_w=crop_size) 
        imageio.imwrite(os.path.join(cropped_dir, bname), crop_img) 

    max_box = 0
    name2center = {}
    index2center = {}
    for idx, i in enumerate(rgb_img_list):     
        img = imageio.imread(i)
        bname = os.path.basename(i)
            
        box_center = get_crop_center(img)
        name2center['{}_{:02d}'.format(obj_name, int(bname[0:3]))] = box_center
        index2center[int(bname[0:3])] = box_center
        print(f'instance {idx} center = ', box_center)
    np.save('{}/{}_center_crop_size_{}.npy'.format(out_dir, obj_name, str(crop_size)), index2center)
    print('Done stage 0: mask and crop!')