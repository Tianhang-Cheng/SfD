import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
np.set_printoptions(suppress=True)

import argparse

import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm
from termcolor import colored

from utils.rend_util import draw_p
from utils.pose_transform import blender_to_opencv

def points_world_to_camera(n, points_3d_cano: List, vc_pred_inv):

    """
    vc_pred_inv is world to camera
    """

    assert len(points_3d_cano) == n
    assert len(vc_pred_inv) == n
    
    points_3d_vc = [[] for _ in range(n)] 

    for i in range(n):
        p = points_3d_cano[i]
        p = (vc_pred_inv[i] @ p.T).T
        p[:, 0:3] = p[:, 0:3] / p[:, 3:4]
        points_3d_vc[i] = p[:, 0:3] 

    res = {'points_3d_vc': points_3d_vc}

    return res

def save_depth_image(h, w, points_2d, depth, instance_dir, use_color_bar=False, postfix=''):
    img = np.zeros((h, w))
    # points_2d[:,0] = np.clip(points_2d[:,0], 0,h-1)
    # points_2d[:,1] = np.clip(points_2d[:,1], 0,w-1)
    img[points_2d[:,1].astype(int), points_2d[:,0].astype(int)] = depth  
    my_dpi = 120
    plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
    im = plt.imshow(img, cmap='gray')
    scatter = plt.scatter(points_2d[:,0].astype(int), points_2d[:,1].astype(int), c=depth, cmap='jet', s=50, edgecolors='none')
    if use_color_bar:
        cbar = plt.colorbar(scatter, ticks=np.linspace(np.min(depth), np.max(depth), 8))  
        plt.savefig(os.path.join(instance_dir, 'depth_sfm_bar_{}.png'.format(postfix)))
    else:
        plt.axis('off')
        plt.savefig(os.path.join(instance_dir, 'depth_sfm_nobar_{}.png'.format(postfix)), bbox_inches='tight', pad_inches=0)
    plt.close() 

def get_scales_from_pose(M):
    """
    M [b,4,4]
    return [3]
    """
    rot = M[:, 0:3, 0:3]
    scale = np.linalg.norm(rot, ord=2, axis=1) 
    return scale
  
ref_index = 0
to_neus = True

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()   
    parser.add_argument(
        '--train_res', type=int, default=800, help='training resolution for nerf')
    parser.add_argument(
        '--instance_num', type=int, default=7, required=True, help='number of instances in the image')
    parser.add_argument(
        '--instance_dir', type=str, default=None, help='instance directory'
    )
    args = parser.parse_args()

    ########################### points ###########################

    # read data info
    n = same_obj_num = args.instance_num
    train_res = args.train_res
    instance_dir = args.instance_dir

    # read path

    raw_dir = os.path.join(instance_dir, 'raw')
    temp_dir = os.path.join(raw_dir, 'temp')
    sfm_outputs_dir = os.path.join(temp_dir, 'sfm_outputs')

    output_path = instance_dir

    # read 3d points cloud
    point_path = os.path.join(sfm_outputs_dir, 'points3D.txt')
    points_list = {}
    with open(point_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            info = line.split(' ')
            p3d = [float(info[1]), float(info[2]), float(info[3]), 1.0, float(info[7])] # 7 is error
            points_list[int(info[0])] = p3d
    
    # read 2d points of images
    image_path = os.path.join(sfm_outputs_dir, 'images.txt')
    points_3d_cano = [[] for _ in range(n)] 
    points_2d = [[] for _ in range(n)]
    points_error = [[] for _ in range(n)]
    with open(image_path, 'r') as f:
        lines = f.readlines()
        lines = lines[1:]
        for i in tqdm(range(len(lines) // 2)): 
            img_id = int(lines[2*i].split(' ')[0]) - 1 # 第一行是图片的编号
            ps = lines[2*i+1].split(' ') # 第二行是图片对应的2d点信息
            for j in range(len(ps)//3):
                # 每个点是(x,y,3d_id)
                if int(ps[3*j+2]) != -1:
                    points_2d[img_id].append([float(ps[3*j]), float(ps[3*j+1])])
                    points_3d_cano[img_id].append(points_list[int(ps[3*j+2])][0:4])  
                    points_error[img_id].append(points_list[int(ps[3*j+2])][4])  
    
    empty_indexes = [index for index, sublist in enumerate(points_2d) if not sublist]
    non_empty_indexes = [index for index, sublist in enumerate(points_2d) if sublist]

    points_2d = [sublist for sublist in points_2d if sublist]
    points_error = [sublist for sublist in points_error if sublist]
    points_3d_cano = [sublist for sublist in points_3d_cano if sublist]

    points_2d = [np.array(x) for x in points_2d]
    points_error = [np.array(x) for x in points_error]
    points_3d_cano = [np.array(x) for x in points_3d_cano]

    # large_error = [len(k) for k in points_2d]
    # large_error_index = large_error < np.mean(large_error) * 0.5
    # large_error_index = np.nonzero(large_error_index)[0]
    # empty_indexes += np.array(non_empty_indexes)[large_error_index].tolist()
    # delete_indexes = large_error_index
    # non_empty_indexes = [x for x in non_empty_indexes if x not in empty_indexes]

    # if len(points_3d_cano) == len(non_empty_indexes):
    #     # filter
    #     points_3d_cano = [points_3d_cano[i] for i in range(n) if i in non_empty_indexes]
    #     points_2d = [points_2d[i] for i in range(n) if i in non_empty_indexes]
    #     points_error = [points_error[i] for i in range(n) if i in non_empty_indexes]
    # else:
    #     # filter
    #     points_3d_cano = [element for index, element in enumerate(points_3d_cano) if index not in delete_indexes]
    #     points_2d = [element for index, element in enumerate(points_2d) if index not in delete_indexes]
    #     points_error = [element for index, element in enumerate(points_error) if index not in delete_indexes] 

    for i in range(len(non_empty_indexes)):
        assert points_2d[i].shape[0] == points_3d_cano[i].shape[0]
    points_2d = np.concatenate(points_2d, axis=0)
    points_error = np.concatenate(points_error, axis=0) 
    points_3d_cano_cat = np.concatenate(points_3d_cano, axis=0) 

    l = [0]+[len(k) for k in points_3d_cano]
    l = np.cumsum(l)
    
    ########################### camera poses ###########################

    cam_path = os.path.join(sfm_outputs_dir, 'poses.npz')
    f_cam = np.load(cam_path, allow_pickle=True)['arr_0'].item()
        
    if to_neus:
        points_3d_cano = [x * np.array([1,-1,-1,1]) for x in points_3d_cano] 
    
    # read sfm camera poses
    vc_pred_inv = []
    for i in range(len(non_empty_indexes)):
        img_name = '{}_rgb.png'.format(str(non_empty_indexes[i]).zfill(3))
        vc_pred_inv.append(f_cam[img_name])
    vc_pred_inv = np.stack(vc_pred_inv, axis=0) # vc <- w 
    if to_neus:
        # to neus format
        vc_pred_inv = blender_to_opencv(vc_pred_inv)
    vc_pred = np.linalg.inv(vc_pred_inv) # c -> w

    # (canonical) world space to virtual camera space 
    res = points_world_to_camera(n=len(non_empty_indexes), points_3d_cano=points_3d_cano, vc_pred_inv=vc_pred_inv) 
    points_3d_vc = np.concatenate(res['points_3d_vc'], axis=0) # real camera space, vc space 

    # suppose real camera = vc[ref]
    # save point cloud of the whole scene and each instance in world space
    c = vc_pred[ref_index]
    points_world = (c[0:3, 0:3] @ points_3d_vc.T).T + c[0:3, 3]
    points_save = {'points_world_all': points_world}
    for i in range(len(non_empty_indexes)):
        points_save.update({'points_world_{}'.format(i):points_world[l[i]:l[i+1]]}) 
    np.save('{}/points_world'.format(output_path), points_save) 
    # draw_p(np.concatenate([points_world[:, 0:3], c[None,0:3, 3]], axis=0))

    # save the index of valid instances
    np.savetxt(os.path.join(output_path, 'non_empty_indexes.txt'), non_empty_indexes) 

    # get and save worinigal depth, here depth is distance along the ray_d direction
    depth = np.linalg.norm(points_3d_vc, ord=2, axis=1)
    save_depth_image(train_res, train_res, points_2d, depth, instance_dir=instance_dir, use_color_bar=True, postfix='origin')

    # draw world points  
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'gray', 'dimgray', 'brown', 'saddlebrown', 'orange', 'darkorange', 'purple', 'indigo']
    colors = colors * (len(non_empty_indexes) // len(colors) + 1)
    for i in range(len(non_empty_indexes)):
        ax.scatter(points_3d_vc[l[i]:l[i+1], 0],
                    points_3d_vc[l[i]:l[i+1], 1],
                    points_3d_vc[l[i]:l[i+1], 2], c=colors[i]) 
    plt.show()


    # save depth
    # points_uv = points_2d 
    # np.save('{}/{}/train/000_depth_uv'.format(cur_path, obj_name), points_uv)
    # np.save('{}/{}/train/000_depth_sfm'.format(cur_path, obj_name), np.stack([depth, points_error], axis=1))

    print(colored('Stage 6:Extract sfm point cloud done!', 'green'))
