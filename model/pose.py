import sys
sys.path.append('../Dup')

import torch
import torch.nn as nn

import os
import json

import numpy as np
from enum import Enum 
import matplotlib.pyplot as plt
from easydict import EasyDict as edict

from utils.rotation_conversions import *
from utils.metrics import rotation_distance_torch
from utils.rend_util import draw_p
from utils.visualize_camera import plot_save_poses_simple

def exist_gt_pose(data_split_dir):
    path = os.path.join(data_split_dir, 'blender_object_gt_pose.json')
    return os.path.exists(path=path)

def exist_sfm_pose(data_split_dir):
    path = os.path.join(data_split_dir, 'object_pred_pose.json')
    return os.path.exists(path=path)

def load_gt_pose(same_obj_num, data_split_dir, verbose=False, non_empty_index=None):
    path = os.path.join(data_split_dir, 'blender_object_gt_pose.json')
    if verbose:
        print('Load gt pose from {}'.format(path)) 
    
    with open(path, 'r') as f:
        meta = json.load(f)  

    key_word = list(meta)[0].split('_')[0]

    if non_empty_index is not None:
        valid_length = len(non_empty_index)
        assert valid_length <= same_obj_num
        obj_poses_gt = np.zeros([valid_length, 4, 4], dtype=np.float32)
        for i in range(valid_length):
            obj_poses_gt[i] = meta['{0}_{1}'.format(key_word, str(non_empty_index[i]).zfill(2))]
        obj_poses_gt = torch.tensor(obj_poses_gt).float().cuda()
    else:
        obj_poses_gt = np.zeros([same_obj_num, 4, 4], dtype=np.float32)
        for i in range(same_obj_num):
            obj_poses_gt[i] = meta['{0}_{1}'.format(key_word, str(i).zfill(2))]
        obj_poses_gt = torch.tensor(obj_poses_gt).float().cuda()
    return obj_poses_gt

def load_sfm_pose(same_obj_num, data_split_dir, verbose=False, non_empty_index=None):
    path = os.path.join(data_split_dir, 'object_pred_pose.json')
    if verbose: 
        print('Load sfm pose from {}'.format(path)) 

    with open(path, 'r') as f:
        meta = json.load(f)   
    
    key_word = list(meta)[0].split('_')[0]

    if non_empty_index is not None:
        valid_length = len(non_empty_index)
        assert valid_length <= same_obj_num
        assert len(meta.keys()) == valid_length
        obj_poses_init = torch.zeros([valid_length, 4, 4], requires_grad=False)
        for i in range(valid_length):
            obj_poses_init[i] = torch.tensor(meta['{0}_{1}'.format(key_word, str(i).zfill(2))])
    else:
        obj_poses_init = torch.zeros([same_obj_num, 4, 4], requires_grad=False)
        for i in range(same_obj_num):
            obj_poses_init[i] = torch.tensor(meta['{0}_{1}'.format(key_word, str(i).zfill(2))])

    return obj_poses_init.float().cuda()

def load_scale_mat(data_split_dir):
    path = os.path.join(data_split_dir, 'object_scale_matrix.json')
    with open(path, 'r') as f:
        meta = json.load(f) 
    scale_mat = torch.tensor(meta['scale_matrix'])
    return scale_mat

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

def get_translation_from_matrix(M):
    """
    M [b,4,4]

    return [b,3,1]
    """
    t = M[:,0:3,3:4].clone() 
    return t.float()

def get_scale_from_pose(M):
    """
    M [b,4,4]
    return [3]
    """
    rot = M[:, 0:3, 0:3]
    scale = torch.norm(rot, p=2, dim=1) 
    return scale

def procrustes_analysis(X0, X1): # [N,3]
    """
    0 is gt
    1 is pred
    """
    # translation
    t0 = X0.mean(dim=0,keepdim=True)
    t1 = X1.mean(dim=0,keepdim=True)
    X0c = X0 - t0
    X1c = X1 - t1
    # scale
    s0 = (X0c ** 2).sum(dim=-1).mean().sqrt()
    s1 = (X1c ** 2).sum(dim=-1).mean().sqrt()
    X0cs = X0c / s0
    X1cs = X1c / s1
    # rotation (use double for SVD, float loses precision)
    U, _, V = (X0cs.t() @ X1cs).double().svd(some=True) 
    R = (V @ U.t()).float()
    if R.det() < 0:
        V[-1] *= -1
        R = (V @ U.t()).float()
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    sim3 = edict(t0=t0[0], t1=t1[0], s0=s0, s1=s1, R=R)
    return sim3
 

def prealign_cameras(pred, gt, verbose=False):

    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred).float().cuda()
    if not isinstance(gt, torch.Tensor):
        gt = torch.tensor(gt).float().cuda()

    pred = pred.clone()
    gt = gt.clone()

    T_pred = pred[:, 0:3, 3] # [N,3]
    T_gt = gt[:, 0:3, 3] # [N,3]

    R_gt = gt[:, 0:3, 0:3]
    R_pred = pred[:, 0:3, 0:3]

    try:
        sim3 = procrustes_analysis(T_gt, T_pred)
    except:
        print("warning: SVD did not converge...")
        import pdb
        pdb.set_trace()
        sim3 = edict(t0=0, t1=0, s0=1, s1=1, R=torch.eye(3).cuda())
    
    if verbose:
        print('t0 = {}, t1 = {}, s0 = {}, s1 = {}'.format(sim3.t0, sim3.t1, sim3.s0, sim3.s1))

    # align the camera poses
    T_pred = (T_pred - sim3.t1) / sim3.s1 @ sim3.R * sim3.s0 + sim3.t0 
    # draw_p(center_gt.detach().cpu(), t_aligned.detach().cpu())

    # align rotation
    U, S, Vh = torch.linalg.svd((R_pred @ R_gt.transpose(2, 1)).sum(0)) 
    R_calibrator = U @ Vh 
    if torch.det(R_calibrator) < 0:
        U[:, -1] *= -1  # Flip the sign of the last column of U
        R_calibrator = U @ Vh
    # print(rotation_distance_torch(R_calibrator @ R_pred, R_gt, degrees=True)) 
    # pose_aligned = concat_to_6D_matrix(R_calibrator @ R_pred, t_aligned[..., None])
    pred_aligned = concat_to_6D_matrix(R_pred, T_pred[..., None]) 
    return pred_aligned

def evaluate_camera_alignment(pose_aligned, pose_GT, degress):

    if not isinstance(pose_aligned, torch.Tensor):
        pose_aligned = torch.tensor(pose_aligned).float().cuda()
    if not isinstance(pose_GT, torch.Tensor):
        pose_GT = torch.tensor(pose_GT).float().cuda()
    # measure errors in rotation and translation
    R_aligned, t_aligned = pose_aligned.split([3,1], dim=-1)
    R_GT, t_GT = pose_GT.split([3,1], dim=-1)
    R_error = rotation_distance_torch(R_aligned, R_GT, degrees=degress)
    t_error = (t_aligned - t_GT)[..., 0:3, 0].norm(dim=-1) # we should compare 3d vector rather than homo vector
    error = edict(R=R_error,t=t_error)
    return error


Init_list = ['GT', 'SFM', 'SFM_noise']
Init = Enum('Choice', Init_list)

class ObjectPose(nn.Module):
    def __init__(self,
                 same_obj_num,
                 visible_num=-1,
                 data_split_dir='', 
                 real_world=False,
                 init_method:str='GT',
                 **kwargs):
        super().__init__()  

        self.has_gt_pose = exist_gt_pose(data_split_dir=data_split_dir)
        self.has_sfm_pose = exist_sfm_pose(data_split_dir=data_split_dir)
        self.train_pose = kwargs.get('train_pose', False)

        self.non_empty_index = np.loadtxt(os.path.join(data_split_dir, 'non_empty_indexes.txt')).astype(int)

        self.same_obj_num = same_obj_num 
        self.real_world = real_world
        self.synthetic = not real_world
        assert visible_num <= len(self.non_empty_index)
        print('visible: {}, valid: {}, total: {}'.format(visible_num, len(self.non_empty_index), same_obj_num))
        self.visible_num = visible_num
        if self.synthetic:
            assert self.has_gt_pose, 'No gt pose found.'
        
        variant_vector_size = kwargs.get('variant_vector_size', 0)
        self.variant_vector_size = variant_vector_size
        if variant_vector_size > 0:
            self.variant_vector = nn.Parameter(torch.randn([self.same_obj_num, variant_vector_size]).cuda() * 0.01, requires_grad=True) 

        # which method to get init pose  
        init_method = self.check_init_method(init_method)
        self.init_method = init_method
        if init_method == Init['GT']:
            print('Use gt pose.')
            self.use_gt = True
        elif init_method == Init['SFM']:
            print('Use sfm pose.')
            self.use_gt = False

        # load object pose
        if init_method == Init['GT'] or self.synthetic:
            self.gt = load_gt_pose(same_obj_num=same_obj_num, data_split_dir=data_split_dir, non_empty_index=self.non_empty_index).cuda()
            assert torch.allclose(get_scale_from_pose(self.gt), torch.ones([self.gt.shape[0], 3]).cuda()), 'GT pose should be orthogonal matrix'
            self.gt_inv = torch.linalg.inv(self.gt) 

        if init_method in [Init['SFM'], Init['SFM_noise']]:
            self.sfm = load_sfm_pose(same_obj_num=same_obj_num, data_split_dir=data_split_dir, non_empty_index=self.non_empty_index).cuda()
            assert torch.allclose(get_scale_from_pose(self.sfm), torch.ones([self.sfm.shape[0], 3]).cuda()), 'Predicted pose should be orthogonal matrix' 
            # self.sfm_inv = torch.linalg.inv(self.sfm)
            self.scale_mat = load_scale_mat(data_split_dir=data_split_dir).cuda()
            self.scale_mat_inv = torch.linalg.inv(self.scale_mat)

            self.init_q = matrix_to_quaternion(self.sfm[:, 0:3, 0:3]).clone()
            self.init_t = self.sfm[:, 0:3, 3:4].clone()  
            if init_method == Init['SFM_noise']:
                q_noise = torch.rand_like(self.init_q).cuda() * 0.03
                t_noise = torch.rand_like(self.init_t).cuda() * 0.03
            else:
                q_noise = 0
                t_noise = 0

            if self.train_pose:
                self.object_q = nn.Parameter(self.init_q.clone() + q_noise, requires_grad=True) # hard to optimize
                self.object_t = nn.Parameter(self.init_t.clone() + t_noise, requires_grad=True) # True, False
            else:
                self.object_q = nn.Parameter(self.init_q.clone() + q_noise, requires_grad=False) # hard to optimize
                self.object_t = nn.Parameter(self.init_t.clone() + t_noise, requires_grad=False) # True, False 
        
        # load gt camera pose and sfm camera to evaluate the prediction in training time

        if self.synthetic:
            json_path = '{}/blender_camera_gt_pose.json'.format(data_split_dir) # use original real camera pose 
            with open(json_path, 'r') as fp:
                meta = json.load(fp)
            c = np.array(meta['frames'][0]['transform_matrix'])  # c2w, real camera poses, blender coordinate
            self.blender_c = torch.tensor(c).float().cuda()

        json_path = '{}/transforms_train.json'.format(data_split_dir) # use original real camera pose
        with open(json_path, 'r') as fp:
            meta = json.load(fp)
        sfm_c = np.array(meta['frames'][0]['transform_matrix'])  # c2w, real camera poses
        self.sfm_c = torch.tensor(sfm_c).float().cuda()
        self.sfm_c_inv = torch.linalg.inv(self.sfm_c)

    
    def check_init_method(self, s):
        if s not in Init_list:
            raise LookupError
        init_method = Init[s]
        if init_method in [Init.GT] and not self.has_gt_pose:
            raise ValueError('No gt pose configuration found.')
        if init_method == Init.SFM and not self.has_sfm_pose:
            raise ValueError('No pose configuration found.')
        return init_method

    def get_gt_pose(self):
        return self.gt[0:self.visible_num]
    
    def get_gt_inv_pose(self): 
        return self.gt_inv[0:self.visible_num]

    def get_pose(self, enable_scale=True):
        """
        set enable_scale to True to compare with GT pose
        get object to world pose
        """
        if self.init_method == Init.GT:
            return self.get_gt_pose()
        else: 
            r = quaternion_to_matrix(self.object_q)
            t = self.object_t
            object_pose = concat_to_6D_matrix(rotation_matrix=r, translation_vector=t) 
            if enable_scale:
                object_pose = object_pose @ self.scale_mat
            return object_pose[0:self.visible_num]
    
    def get_inv_pose(self, enable_scale=True):
        """
        get world to object (canonical/local space) pose
        """
        if self.init_method == Init.GT: 
            return self.get_gt_inv_pose() 
        else:
            q_inv = quaternion_invert(self.object_q)
            r_inv = quaternion_to_matrix(q_inv)
            t = self.object_t
            object_pose = concat_to_6D_matrix(rotation_matrix=r_inv, translation_vector=-torch.bmm(r_inv, t)) 
            if enable_scale:
                object_pose = self.scale_mat_inv @ object_pose
            return object_pose[0:self.visible_num] 

    def get_variant_vector(self):
        if self.variant_vector_size > 0:
            return self.variant_vector[0:self.visible_num]
        return None

    def get_var_of_variant_vector(self):
        vectors = self.get_variant_vector()
        if vectors is None:
            return torch.tensor(0.0).cuda()
        return torch.var(vectors, dim=1).mean(0)
    
    def get_diff(self, degress=True, image_path=None): 
        """
        image_path: the path of the image to draw the camera pose, if None, do not draw
        return:
            dr [n]
            dt [n]
        """ 

        dr = torch.zeros([self.visible_num]).cuda()
        dt = torch.zeros([self.visible_num]).cuda() 

        # transform to multi-camera single-object format
        if self.has_gt_pose:
            
            pred_eval = self.get_inv_pose(enable_scale=False) @ self.sfm_c
            gt_eval = self.gt[0] @ self.gt_inv @ self.blender_c[None].expand(pred_eval.shape[0], -1, -1) # gt_eval is fixed for different experiments

            pred_eval = pred_eval.detach().cpu().numpy()
            gt_eval = gt_eval.detach().cpu().numpy()
            
            # normalize
            _pred_eval = np.linalg.inv(pred_eval[0]) @ pred_eval
            _gt_eval = np.linalg.inv(gt_eval[0]) @ gt_eval

            _pred_eval_center = _pred_eval[:, 0:3, 3].mean(0)
            _pred_eval[:, 0:3, 3] = _pred_eval[:, 0:3, 3] - _pred_eval_center
            _pred_eval_length = np.linalg.norm(_pred_eval[:, 0:3, 3], ord=2, axis=1).mean(0)
            _pred_eval[:, 0:3, 3] = _pred_eval[:, 0:3, 3] / _pred_eval_length

            _gt_eval_center = _gt_eval[:, 0:3, 3].mean(0)
            _gt_eval[:, 0:3, 3] = _gt_eval[:, 0:3, 3] - _gt_eval_center
            _gt_eval_length = np.linalg.norm(_gt_eval[:, 0:3, 3], ord=2, axis=1).mean(0)
            _gt_eval[:, 0:3, 3] = _gt_eval[:, 0:3, 3] / _gt_eval_length

            # calculate error
            _pred_eval = prealign_cameras(_pred_eval, _gt_eval, verbose=False)
            error = evaluate_camera_alignment(_pred_eval, _gt_eval, degress=True)
            dr = error.R
            dt = error.t
            if image_path is not None:
                # geo_length = np.linalg.norm(points.max(0) - points.min(0))
                plot_save_poses_simple(
                    pose=_pred_eval.detach().cpu().numpy(),
                    pose_ref=_gt_eval,
                    points=None, 
                    path=image_path,
                    scale=1,
                    cam_depth_scale=3,
                    show=False,
                    camera_look_at_positive_z=False)
            
        return dr, dt
    
if __name__ == '__main__':
    # obj_poses = ObjectPose(same_obj_num=10, init_method='SFM', visible_num=10, real_world=False, data_split_dir=r'E:\dataset\DuplicateSingleImage\box', train_pose=True)
    # obj_poses = ObjectPose(same_obj_num=70, init_method='SFM', visible_num=70, real_world=False, data_split_dir=r'E:\dataset\DuplicateSingleImage\paint', train_pose=True)
    obj_poses = ObjectPose(same_obj_num=70, init_method='SFM', real_world=False, data_split_dir=r'E:\dataset\DuplicateSingleImage\paint', train_pose=True)
    dr, dt = obj_poses.get_diff(degress=True)
    print('# instance = {}, dr = {} degree, dt = {}'.format(0, torch.median(dr).item(), torch.median(dt).item())) # print(dr.mean().item(), dt.mean().item()) 