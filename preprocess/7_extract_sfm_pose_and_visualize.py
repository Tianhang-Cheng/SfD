
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
np.set_printoptions(suppress=True)

import json
import socket
import torch
import argparse
import shutil

import matplotlib.pyplot as plt
from termcolor import colored

from utils.point_cloud import keep_points_near_center
from utils.rend_util import draw_p
# from utils.visualize_camera import vis_cameras, plot_save_poses_simple
# from utils.metrics import rotation_distance_numpy, rotation_distance_torch
from utils.pose_transform import blender_to_opencv, check_rotation_scale
# from model.pose import load_gt_pose, prealign_cameras, evaluate_camera_alignment

def concat_to_6D_matrix(rotation_matrix, translation_vector):
    """
    Input
        rotation_matrix [b,3,3]
        translation_vector [b,3,1]
    Return
        M [b, 4, 4]
    """
    batch = rotation_matrix.shape[0]

    homo_vector = torch.tensor([0,0,0,1])[None, None, :].expand(batch,-1,-1).cuda()

    M = torch.cat([rotation_matrix, translation_vector], dim=2) # [b,3,4]
    M = torch.cat([M, homo_vector], dim=1)

    return M
def get_scales_from_pose(M):
    """
    M [b,4,4]
    return [3]
    """
    rot = M[:, 0:3, 0:3]
    scale = np.linalg.norm(rot, ord=2, axis=1) 
    return scale

def remove_scale_from_pose(pose, inv_scale_mat): 
    """
    pose: [n,4,4]
    """ 
    r = remove_scale_from_rotation(pose[:, 0:3, 0:3], inv_scale_mat) 
    t = pose[:, 0:3, 3:4]
    return concat_to_6D_matrix(r, t) 

def remove_scale_from_rotation(rot, inv_scale_mat): 
    """
    pose: [n,3,3]
    """ 
    r = torch.einsum('nxy, yz -> nxz', rot, inv_scale_mat) 
    return r 

def get_near_far(radius, center, rays_o, rays_d): 
    d_norm = np.sum(rays_d ** 2, axis=-1, keepdims=True)  
    c_to_o = center - rays_o # from camera origin point to object center
    proj = np.sum(c_to_o * rays_d, axis=-1, keepdims=True)
    mid =  proj / d_norm
    near = mid - radius
    far = mid + radius
    return near, far

def sdf(xyz, r=1):
    # xyz [n, 3]
    return np.linalg.norm(xyz, ord=2, axis=1) - r
    
def save_depth_image(h, w, points_2d, depth, postfix=''):
    img = np.zeros((h, w))
    # points_2d[:,0] = np.clip(points_2d[:,0], 0,h-1)
    # points_2d[:,1] = np.clip(points_2d[:,1], 0,w-1)
    img[points_2d[:,1].astype(int), points_2d[:,0].astype(int)] = depth  
    my_dpi = 120
    plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
    im = plt.imshow(img, cmap='gray')
    scatter = plt.scatter(points_2d[:,0].astype(int), points_2d[:,1].astype(int), c=depth, cmap='jet', s=50, edgecolors='none')
    cbar = plt.colorbar(scatter, ticks=np.linspace(np.min(depth), np.max(depth), 8))  
    plt.axis('off')
    plt.savefig('depth_sfm.png'.format(processed_data_path, obj_name, postfix), bbox_inches='tight', pad_inches = 0 )
    plt.close() 

if __name__ == "__main__": 

    ########################### input ########################### 

    parser = argparse.ArgumentParser()   
    parser.add_argument(
        '--train_res', type=int, default=800, help='training resolution for nerf')
    parser.add_argument(
        '--instance_num', type=int, default=7, required=True, help='number of instances in the image')
    parser.add_argument(
        '--instance_dir', type=str, default=None, help='instance directory'
    )
    args = parser.parse_args()

    ref_index = 0
    dump_pose = True
    
    ########################### read ########################### 
    
    # read object information
    # read data info
    n = same_obj_num = args.instance_num
    train_res = args.train_res
    instance_dir = args.instance_dir

    # read path
    raw_dir = os.path.join(instance_dir, 'raw')
    temp_dir = os.path.join(raw_dir, 'temp')
    sfm_outputs_dir = os.path.join(temp_dir, 'sfm_outputs')
    real_world = True
    
    # define path
    output_dir = instance_dir
    train_dir = os.path.join(instance_dir, 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    cam_intrinsic_path = os.path.join(sfm_outputs_dir, 'cameras.txt')
    cam_path = os.path.join(sfm_outputs_dir, 'poses.npz')

    # read valid index
    non_empty_index_path = os.path.join(instance_dir, 'non_empty_index.txt')
    if os.path.exists(non_empty_index_path):
        non_empty_index = np.loadtxt(non_empty_index_path).astype(int)
    else:
        non_empty_index = np.arange(0, n).astype(int)

    # read points cloud
    points_world = np.load(os.path.join(instance_dir, 'points_world.npy'), allow_pickle=True).item()
    points_world_ref = keep_points_near_center(points_world['points_world_0'], keep_ratio=0.99) # point is already neus coordinate
    
    # read camera pose from sfm result
    meta = np.load(cam_path, allow_pickle=True)['arr_0'].item()
    vc_pred_inv = [] #  world to camera
    for i in range(len(non_empty_index)):
        img_name = '{}_rgb.png'.format(str(non_empty_index[i]).zfill(3)) 
        vc_pred_inv.append(meta[img_name])
    vc_pred_inv = np.stack(vc_pred_inv, axis=0) # world to v camera
    vc_pred_inv = blender_to_opencv(vc_pred_inv) # blender
    vc_pred = np.linalg.inv(vc_pred_inv)  # vc -> w 
    
    center = points_world_ref.mean(0)
    radius = 0.5 * np.linalg.norm(np.max(points_world_ref, axis=0) - np.min(points_world_ref, axis=0)) 
    scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    scale_mat[:3, 3] = center # np.linalg.inv(scale_mat) is from obj space to a NDC space  

    cx = train_res // 2
    cy = train_res // 2

    f = open(cam_intrinsic_path, 'r') 
    data = f.readlines()[1]
    focal = data.split(' ')[4]

    fx = fy = float(focal)
    sfm_c = vc_pred[ref_index] # cam to world, sfm camera pose in neus coordinate
    sfm_c_inv = np.linalg.inv(sfm_c) # world to cam

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    # read points cloud

    point_path = os.path.join(sfm_outputs_dir, 'points3D.txt')
    points = []
    with open(point_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            info = line.split(' ')
            points.append([float(info[1]), float(info[2]), float(info[3])])

    points = np.array(points)
    points_center = points.mean(0)
    points = points - points_center # center points to origin, for better visualization
    # draw_p(points) # share instance
    print('Total points = {}'.format(len(points))) 

    ########################### dump poses ########################### 
    """
    vc_pred is C, camera to world
    obj_pose_pred is O, object to world
    obj_pose_pred_inv is O ^ -1
    we find O ^ -1 = O0 ^ - 1 @ Ci @ C0 ^ - 1
    => O0 @ O ^ -1 = Ci @ C0 ^ - 1
    => (O @ O0 ^ -1) = Ci @ C0 ^ - 1
    """
    # obj_pose_pred_scaled = vc_pred[ref_index] @ vc_pred_inv @ scale_mat # obj_pose_pred # object_to_world
    obj_pose_pred = vc_pred[ref_index] @ vc_pred_inv  # obj_pose_pred # object_to_world
    obj_pose_pred_inv = np.linalg.inv(obj_pose_pred) # world to object
    check_rotation_scale(obj_pose_pred)

    if dump_pose:

        # write object scale

        with open(instance_dir + '/object_scale_matrix.json', 'wt') as json_file:
            json.dump({'scale_matrix': scale_mat.tolist()}, json_file, indent=4, separators=(",", ": "))
        print(colored('Dumpling object scale matrix to {}'.format(os.path.join(instance_dir, 'object_scale_matrix.json')), 'magenta', attrs=['bold']))

        # write object pose
        # import pdb; pdb.set_trace()
        object_name = os.path.basename(instance_dir)

        obj_meta_data = {} 
        for i in range(len(non_empty_index)):      
            obj_meta_data['{}_{}'.format(object_name ,str(i).zfill(2))] = obj_pose_pred[i]
        with open(instance_dir + '/object_pred_pose.json', 'wt') as json_file:
            json.dump({k: v.tolist() for k, v in obj_meta_data.items()}, json_file, indent=4, separators=(",", ": "))
        print(colored('Dumpling pred object-to-world pose to {}'.format(os.path.join(instance_dir, 'object_pred_pose.json')), 'magenta', attrs=['bold'])) 

        # write camera pose

        camera_matrix_list = sfm_c.tolist()
        file_path = os.path.join(instance_dir, 'transforms_train.json')
        with open(file_path, "w") as json_file:
            json.dump({"focal": focal,
                        "frames": [{"file_path": "train/000",
                                    "transform_matrix": camera_matrix_list}]}, json_file, indent=4, separators=(",", ": ")) 
        file_path = os.path.join(instance_dir, 'transforms_test.json')
        with open(file_path, "w") as json_file:
            json.dump({"focal": focal,
                        "frames": [{"file_path": "test/000",
                                    "transform_matrix": camera_matrix_list}]}, json_file, indent=4, separators=(",", ": ")) 
        print(colored('Dumpling pred camera-to-world pose to {}'.format(os.path.join(instance_dir, 'transforms_train.json')), 'magenta', attrs=['bold']))
        print(colored('Dumpling pred camera-to-world pose to {}'.format(os.path.join(instance_dir, 'transforms_test.json')), 'magenta', attrs=['bold'])) 

    ########################### debug ########################### 

    # get points in object space (check if it's in unit sphere)

    debug_points = True
    obj_pose_pred_inv_scaled = np.linalg.inv(scale_mat) @ vc_pred @ vc_pred_inv[ref_index]   # scaled world to object. vc_pred is camera to world 
    points_obj = (obj_pose_pred_inv_scaled[0, 0:3, 0:3] @ points_world_ref.T).T + obj_pose_pred_inv_scaled[0, 0:3, 3]
    points_obj_all = []
    for i in range(len(non_empty_index)):
        p = (obj_pose_pred_inv_scaled[i, 0:3, 0:3] @ points_world['points_world_{}'.format(i)].T).T + obj_pose_pred_inv_scaled[i, 0:3, 3]
        points_obj_all.append(p)
    points_obj_all = np.concatenate(points_obj_all, axis=0)
    if debug_points:
        draw_p(points_obj)
    
    # render image in Neus coordinate with fake sdf function
    points_world_all = points_world['points_world_all']
    points_camera_all = (sfm_c_inv[0:3, 0:3] @ points_world_all.T).T + sfm_c_inv[0:3, 3]
    # points_sdf_all = sdf(points_obj_all, r=1.5)
    depth = np.linalg.norm(points_camera_all, ord=2, axis=1)
    points_2d = (K @ points_camera_all.T).T
    points_2d = points_2d[:, 0:2] / points_2d[:, 2:3]
    # save_depth_image(points_2d, depth, postfix='test')

    h, w = train_res, train_res
    uv = np.mgrid[0:h:30, 0:w:30].astype(np.int32)
    uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
    uv = uv.permute(1, 2, 0)[None]
    uv = uv.reshape(-1, 2)  # [H*W, 2]
    steps = 4  
    x_cam = uv[:, 0]  # [H*W]
    y_cam = uv[:, 1] 
    z_cam = torch.ones_like(x_cam)
    p_uv = torch.stack([x_cam, y_cam, z_cam], dim=1)
    p_camera = (torch.inverse(torch.from_numpy(K).float()) @ p_uv.T).T
    p_camera = p_camera.cpu().numpy()
    p_camera = p_camera * np.array([1, -1, -1]) # to Neus coordinate
    p_world = (sfm_c[0:3, 0:3] @ p_camera.T).T + sfm_c[0:3, 3]
    p_dir = p_world - sfm_c[0:3, 3]
    p_dir = p_dir / np.linalg.norm(p_dir, ord=2, axis=1, keepdims=True)

    # near, far = get_near_far(radius, center, cam[0:3, 3], p_dir)
    # p_3d_world = cam[0:3, 3] +  p_dir[:, None] * ((far - near) * np.linspace(0, 1, steps)[None, :] + near)[..., None]
    # p_3d_world = p_3d_world.reshape(-1, 3)
    # p_3d_cam = (cam_inv[0:3, 0:3] @ p_3d_world.T).T + cam_inv[0:3, 3]

    near, far = min(depth), max(depth)
    p_3d_world = sfm_c[0:3, 3] + p_dir[:, None] * np.linspace(near, far, steps)[None, :, None]
    p_3d_world = p_3d_world.reshape(-1, 3)
    p_3d_cam = (sfm_c_inv[0:3, 0:3] @ p_3d_world.T).T + sfm_c_inv[0:3, 3]

    # draw_p(p_3d_world, y=cam[0:3, 3])
    # draw_p(points_world_all, y=cam[0:3, 3])
    # draw_p(np.concatenate([p_3d_world, points_world_all]), y=cam[0:3, 3])

    ########################### vizualize camera error ########################### 

    # if synthetic: 

    #     # copy real-world camera pose to synthetic dataset
    #     shutil.copy(os.path.join(blender_data_path, obj_name, 'transforms_train.json'), '{}/{}/blender_camera_gt_pose.json'.format(processed_data_path, obj_name))
    #     shutil.copy(os.path.join(blender_data_path, obj_name, 'object_gt_pose.json'), '{}/{}/blender_object_gt_pose.json'.format(processed_data_path, obj_name))

    #     # read real camera pose in synthetic dataset, here is in NeuS coordinate
    #     json_path = '{}/{}/blender_camera_gt_pose.json'.format(processed_data_path, obj_name) # use original real camera pose
    #     print('Read cam from {}'.format(json_path))
    #     with open(json_path, 'r') as fp:
    #         meta = json.load(fp)
    #     blender_c = np.array(meta['frames'][0]['transform_matrix'])  # c2w, real camera poses
    #     blender_c_inv = np.linalg.inv(blender_c) # world to camera 
        
    #     # load gt object pose
    #     blender_o = load_gt_pose(same_obj_num=same_obj_num, data_split_dir='{}/{}'.format(processed_data_path, obj_name), verbose=True).detach().cpu().numpy() # use original bbox, o -> w
    #     blender_o_inv = np.linalg.inv(blender_o) #  w -> o   
    #     blender_cs = blender_o[ref_index] @ blender_o_inv @ blender_c
    #     check_rotation_scale(blender_cs)

    #     # save gt camera pose
    #     # obj_meta_data = {}
    #     # for i in range(n):
    #     #     obj_meta_data['{}_{}'.format(obj_name, str(i).zfill(2))] = camera_pose_gt[i]
    #     # json_text = json.dumps({k: v.tolist() for k, v in obj_meta_data.items()})
    #     # if dump_pose:
    #     #     with open(output_dir + '/camera_gt_pose.json', 'w') as json_file:
    #     #         json_file.write(json_text)
    #     #     print(colored('Dumpling gt world-to-camera pose to {}'.format(os.path.join(output_dir, 'camera_gt_pose.json')), 'magenta', attrs=['bold']))

    #     ########################### evaluate camera ###########################  
        
    #     pred_eval = obj_pose_pred_inv @ sfm_c
    #     gt_eval = blender_o[ref_index] @ blender_o_inv @ blender_c # gt_eval is fixed for different experiments

    #     # normalize
    #     _pred_eval = np.linalg.inv(pred_eval[ref_index]) @ pred_eval
    #     _gt_eval = np.linalg.inv(gt_eval[ref_index]) @ gt_eval

    #     _pred_eval_center = _pred_eval[:, 0:3, 3].mean(0)
    #     _pred_eval[:, 0:3, 3] = _pred_eval[:, 0:3, 3] - _pred_eval_center
    #     _pred_eval_length = np.linalg.norm(_pred_eval[:, 0:3, 3], ord=2, axis=1).mean(0)
    #     _pred_eval[:, 0:3, 3] = _pred_eval[:, 0:3, 3] / _pred_eval_length

    #     _gt_eval_center = _gt_eval[:, 0:3, 3].mean(0)
    #     _gt_eval[:, 0:3, 3] = _gt_eval[:, 0:3, 3] - _gt_eval_center
    #     _gt_eval_length = np.linalg.norm(_gt_eval[:, 0:3, 3], ord=2, axis=1).mean(0)
    #     _gt_eval[:, 0:3, 3] = _gt_eval[:, 0:3, 3] / _gt_eval_length

    #     # calculate error
    #     _pred_eval = prealign_cameras(_pred_eval, _gt_eval, verbose=True)
    #     error = evaluate_camera_alignment(_pred_eval, _gt_eval, degress=True)
    #     print('mean_dr = {} (degress)'.format(error.R.mean()))
    #     print('mean_dt = {} (length)'.format(error.t.mean()))

    #     # save camera pose for visualization in Blender
    #     # os.makedirs('{}/viz_cam'.format(instance_dir), exist_ok=True)
    #     # os.makedirs('{}/viz_cam/save'.format(instance_dir), exist_ok=True)
    #     # np.save('{}/viz_cam/save/gt_cam_{}.npy'.format(instance_dir), blender_cs)
    #     # np.save('{}/viz_cam/save/pred_cam_{}.npy'.format(instance_dir), blender_cs[ref_index] @ np.linalg.inv(pred_eval[ref_index]) @ pred_eval)
    #     # # save relative object pose for visualization in Blender
    #     # np.save('{}/viz_cam/save/gt_obj_{}.npy'.format(instance_dir), blender_o @ blender_o_inv[ref_index])

    #     plot_save_poses_simple(
    #         pose=_pred_eval.detach().cpu().numpy(), # pred_eval  pred_aligned
    #         pose_ref=_gt_eval,
    #         points=None, 
    #         path=os.path.join(instance_dir, 'viz_cam.png'),
    #         scale=1,
    #         cam_depth_scale=3,
    #         show=True,
    #         camera_look_at_positive_z=False,
    #         plot_legend=False)