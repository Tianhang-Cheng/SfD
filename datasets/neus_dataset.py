import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as T

import numpy as np
from termcolor import colored

from datasets.data_info import obj_info
from utils import rend_util
from utils import point_cloud

import json
import imageio
import pdb
import matplotlib.pyplot as plt

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 instance_dir,
                 frame_skip,
                 split='train',
                 same_obj_num=0,
                 visible_num=0,
                 non_empty_indexes=None,
                 select_index:int=-1,
                 real_world=False,
                 use_pretrain_normal=False,
                 ):
        
        print('Creating dataset from: ', instance_dir)
        assert os.path.exists(instance_dir), "Data directory "+instance_dir+" is empty"
        print(colored('Use data split: {}'.format(split), 'red', attrs=['bold']))
        print(colored('This is only for single view experiment', 'red', attrs=['bold'])) #FIXME: only support single view

        self.instance_dir = instance_dir
        self.select_index = select_index
        self.frame_skip = frame_skip
        self.split = split  
        self.real_world = real_world
        self.use_pretrain_normal = use_pretrain_normal
        self.same_obj_num = same_obj_num 
        self.visible_num = visible_num
        self.single_imgname = None
        self.sampling_idx = None

        # load camera poses
        # always use the same camera pose for single view experiment
        json_path = os.path.join(self.instance_dir, 'transforms_train.json')
        # if split == 'test_relight':
        #     json_path = os.path.join(self.instance_dir, 'transforms_test.json')
        # else:
        #     json_path = os.path.join(self.instance_dir, 'transforms_{}.json'.format(split)) 
        with open(json_path, 'r') as fp:
            meta = json.load(fp)
        cam = np.array(meta['frames'][0]['transform_matrix']) # c2w, real camera poses in opencv format
        cam_inv = np.linalg.inv(cam)
        assert len(meta['frames']) == 1, "Only support single view"
        focal = float(meta['focal'])
        # camera_angle_x = float(meta['camera_angle_x'])
        # focal = .5 * img_w / np.tan(.5 * camera_angle_x)

        # load object points from Structure from Motion
        self.scene_center = torch.tensor([0.0, 0.0, 0.0]).cuda().float()
        self.scene_radius = 1.0
        points_path = os.path.join(self.instance_dir, 'points_world.npy')
        points = np.load(points_path, allow_pickle=True).item()
        points_world_all = points['points_world_all']
        points_world_0 = points['points_world_0']
        points_camera_all = (cam_inv[0:3, 0:3] @ points_world_all.T).T + cam_inv[0:3, 3]
        depth = np.linalg.norm(points_camera_all, ord=2, axis=1)
        self.points_world_0 = torch.from_numpy(points_world_0).cuda().float()
        self.near, self.far = np.min(depth), np.max(depth)
        self.scene_center, self.scene_radius = point_cloud.find_points_bbox(points_world_all, keep_ratio=0.95)

        # define path to load images
        image_paths = [] 
        seg_paths = []
        albedos_paths = []
        roughness_paths = []
        cam_poses = []
        normal_paths = []
        pretrain_normal_paths = []
        envmap_b_image_paths = [] 
        envmap_d_image_paths = []
 
        """
        train:          [camera, rgb, seg, normal, mask]
        test:           [camera, rgb, seg, normal, mask, albedo, roughness]
        test_relight:   [camera, rgb, seg]
        """

        for frame in meta['frames']:

            # load camera poses and segmentation images
            cam_poses.append(np.array(frame['transform_matrix']))
            seg_paths.append(os.path.join(self.instance_dir, frame['file_path'] + '_instance_seg.png')) 
            
            # if evaluate relight, load relighting rgb images
            if split == 'test_relight':
                file_path_b = frame['file_path'].replace('train','test').replace('test','test_relight_b')
                file_path_d = frame['file_path'].replace('train','test').replace('test','test_relight_d')
                envmap_b_image_paths.append(os.path.join(self.instance_dir, file_path_b + '_rgb.exr'))
                envmap_d_image_paths.append(os.path.join(self.instance_dir, file_path_d + '_rgb.exr'))
            
            # if training, load training rgb images
            else:
                # load normal images
                normal_paths.append(os.path.join(self.instance_dir, frame['file_path'] + '_normal.png'))
                if self.use_pretrain_normal:
                    pretrain_normal_paths.append(os.path.join(self.instance_dir, frame['file_path'] + '_normal_pretrain.png')) 
                # load rgb images
                if real_world:
                    image_paths.append(os.path.join(self.instance_dir, frame['file_path'] + '_rgb.png')) # real world data format usually is .png
                else:
                    image_paths.append(os.path.join(self.instance_dir, frame['file_path'] + '_rgb.exr'))
                if split == 'test':
                    # load albedo and roughness images
                    albedos_paths.append(os.path.join(self.instance_dir, frame['file_path'] + '_diffuse.exr'))
                    roughness_paths.append(os.path.join(self.instance_dir, frame['file_path'] + '_roughness.exr'))  
            
        # get image size
        if real_world:
            img_h, img_w = rend_util.load_rgb(image_paths[0]).shape[:2]
        elif split == 'test_relight':
            img_h, img_w = rend_util.load_exr(envmap_b_image_paths[0]).shape[:2]
        else:
            img_h, img_w = rend_util.load_exr(image_paths[0]).shape[:2]
        print("Camera info: focal {}, img_w {}, img_h {}".format(focal, img_w, img_h))
        self.img_res = [img_h, img_w]
        self.total_pixels = self.img_res[0] * self.img_res[1] 

        # placeholder data
        self.fake_mask = torch.ones([img_h, img_w]).bool()
        self.fake_image = np.ones([img_h, img_w, 3]) 
        
        # skip data if frame_skip > 1
        cam_poses = self.skip_data(np.array(cam_poses)) 
        seg_paths = self.skip_data(seg_paths) 
        if split == 'test_relight':
            envmap_b_image_paths = self.skip_data(envmap_b_image_paths) 
            envmap_d_image_paths = self.skip_data(envmap_d_image_paths)
            print('Training image: {}'.format(len(envmap_b_image_paths)))
            self.n_cameras = len(envmap_b_image_paths) 
        else:
            image_paths = self.skip_data(image_paths)
            normal_paths = self.skip_data(normal_paths)
            if use_pretrain_normal:
                pretrain_normal_paths = self.skip_data(pretrain_normal_paths)
            if split == 'test':
                albedos_paths = self.skip_data(albedos_paths)
                roughness_paths = self.skip_data(roughness_paths)
            print('Training image: {}'.format(len(image_paths)))
            self.n_cameras = len(image_paths)

        # create camera intrinsics and poses
        self.intrinsics_all = []
        self.pose_all = []
        intrinsics = [[focal, 0, img_w / 2],[0, focal, img_h / 2], [0, 0, 1]]
        intrinsics = np.array(intrinsics).astype(np.float32)
        for i in range(self.n_cameras):
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(cam_poses[i]).float())  

        # load images
        self.rgb_images = []
        self.albedo_images = []
        self.roughness_images = []
        self.object_masks = []
        self.object_segs = []
        self.normal_images = []
        self.pretrain_normal_images = []
        self.relight_rgb_images_b = [] 
        self.relight_rgb_images_d = []

        # read segmentation images
        for path in seg_paths:
            object_seg = rend_util.load_seg(path, input_range='0_255', output_range='0_n', non_empty_indexes=None) # FIXME: non_empty_indexes 
            object_seg = torch.from_numpy(object_seg).to(int)
            object_seg[object_seg > visible_num] = 0 
            self.object_segs.append(object_seg)
            self.object_masks.append((object_seg > 0).bool())

        # read relighting images
        if split == 'test_relight': 
            for path in envmap_b_image_paths:
                rgb = rend_util.load_exr(path) 
                self.relight_rgb_images_b.append(torch.from_numpy(rgb).float()) 
            for path in envmap_d_image_paths:
                rgb = rend_util.load_exr(path) 
                self.relight_rgb_images_d.append(torch.from_numpy(rgb).float())

        # read training or testing images
        else:
            for path in image_paths:
                rgb = rend_util.load_rgb(path) if self.real_world else rend_util.load_exr(path)
                rgb = torch.from_numpy(rgb).float() 
                self.rgb_images.append(rgb)
            for path in normal_paths: 
                normal = self.fake_image if self.real_world else rend_util.load_normal(path)
                self.normal_images.append(torch.from_numpy(normal).float())
            # load pretrain normal images if there is
            if use_pretrain_normal:
                for path in pretrain_normal_paths:
                    normal = rend_util.load_normal(path)  
                    self.pretrain_normal_images.append(torch.from_numpy(normal).float())
            # test images also have albedo and roughness
            if self.split == 'test': 
                for path in albedos_paths: 
                    albedo = self.fake_image if real_world else rend_util.load_rgb(path)
                    self.albedo_images.append(torch.from_numpy(albedo).float())
                for path in roughness_paths:
                    roughness = self.fake_image if real_world else rend_util.load_rgb(path) 
                    self.roughness_images.append(torch.from_numpy(roughness).float()) 
         
        print('\n')

    def skip_data(self, x):
        return [x[self.select_index]] if self.select_index >= 0 else x[::self.frame_skip] 
        
    def __len__(self):
        return (self.n_cameras)

    def __getitem__(self, idx):  

        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx],
            "object_mask": self.object_masks[idx],
            "object_seg": self.object_segs[idx],
        } 

        if self.split == 'train':
            ground_truth = {
                "rgb": self.rgb_images[idx],
                "normal": self.normal_images[idx],
            }  

        if self.split == 'test':
            ground_truth = {
                "rgb": self.rgb_images[idx],
                "normal": self.normal_images[idx],
                "albedo":self.albedo_images[idx],
                "roughness":self.roughness_images[idx],
            }
        
        if self.use_pretrain_normal and self.split != 'test_relight':
            ground_truth["normal_pretrain"] = self.pretrain_normal_images[idx]

        if self.split == 'test_relight':
            ground_truth = {} 

        if self.sampling_idx is not None:
            sample["object_mask"] = self.object_masks[idx][self.sampling_idx]  
            sample["uv"] = uv[self.sampling_idx, :]
            sample["object_seg"] = self.object_segs[idx][self.sampling_idx] 
            if self.split == 'test_relight':
                ground_truth["rgb_b"] = self.relight_rgb_images_b[idx][self.sampling_idx]
                ground_truth["rgb_d"] = self.relight_rgb_images_d[idx][self.sampling_idx]
            else:
                ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx] 
                ground_truth["normal"] = self.normal_images[idx][self.sampling_idx]
            if self.split == 'test':
                ground_truth["albedo"] = self.albedo_images[idx][self.sampling_idx]
                ground_truth["roughness"] = self.roughness_images[idx][self.sampling_idx] 

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, 
        # ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)
    
    def near_far_from_sphere(self, rays_o, rays_d):
        # radius = self.scene_radius
        # d_norm = torch.sum(rays_d**2, dim=-1, keepdim=True) 
        # center = self.scene_center
        # c_to_o = center - rays_o # from camera origin point to object center
        # proj = torch.sum(c_to_o * rays_d, dim=-1, keepdim=True)
        # mid =  proj / d_norm
        # near = mid - radius
        # far = mid + radius
        # return near.float(), far.float()
        near = self.near * torch.ones([rays_o.shape[0], 1]).cuda() * 0.95
        far = self.far * torch.ones([rays_o.shape[0], 1]).cuda() * 1.05
        return near.float(), far.float()
    
    def image_at(self, idx, envmap_name=None, resolution_level=1):
        H, W = self.img_res
        if envmap_name is not None:
            if envmap_name == 'b':
                img = self.relight_rgb_images_b[idx].permute(2,0,1)  
            if envmap_name == 'd':
                img = self.relight_rgb_images_d[idx].permute(2,0,1)
        else:
            img = self.rgb_images[idx].permute(2,0,1) 
        img = T.Resize((H // resolution_level, W // resolution_level))(img).permute(1,2,0)
        return img 
    
    def mask_at(self, idx, resolution_level=1): 
        H, W = self.img_res
        mask = self.object_masks[idx].unsqueeze(0)
        mask = T.Resize((H // resolution_level, W // resolution_level))(mask)[0]
        return mask

    def seg_at(self, idx, resolution_level=1):
        H, W = self.img_res
        seg_gt = self.object_segs[idx].unsqueeze(0)
        seg_gt = T.Resize((H // resolution_level, W // resolution_level))(seg_gt)[0]
        return seg_gt
    
    def normal_at(self, idx, resolution_level=1):
        """
        return shape: [H,W,3]
        value range: [0,1]
        """
        H, W = self.img_res
        normal_gt = self.normal_images[idx].permute(2,0,1) 
        normal_gt = T.Resize((H // resolution_level, W // resolution_level))(normal_gt).permute(1,2,0)
        return normal_gt

    def pretrain_normal_at(self, idx, resolution_level=1):
        """
        return shape: [H,W,3]
        value range: [0,1]
        """
        if self.use_pretrain_normal:
            H, W = self.img_res
            normal_gt = self.pretrain_normal_images[idx].permute(2,0,1) 
            normal_gt = T.Resize((H // resolution_level, W // resolution_level))(normal_gt).permute(1,2,0)
            return normal_gt
        else:
            return None
    
    def rough_at(self, idx, resolution_level=1):
        """
        [H,W,3]
        """
        H, W = self.img_res
        rough_gt = self.roughness_images[idx].permute(2,0,1) 
        rough_gt = T.Resize((H // resolution_level, W // resolution_level))(rough_gt).permute(1,2,0)
        return rough_gt
    
    def albedo_at(self, idx, resolution_level=1):
        """
        [H,W,3]
        """
        H, W = self.img_res
        albedo_gt = self.albedo_images[idx].permute(2,0,1) 
        albedo_gt = T.Resize((H // resolution_level, W // resolution_level))(albedo_gt).permute(1,2,0)
        return albedo_gt
    
    def pose_at(self, idx): 
        return self.pose_all[idx]

    def gen_rays_at(self, img_idx):
        """
        Generate rays at world space from one camera.
        return:
        [1, H*W, 3]
        """ 

        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)  # [2, H, W], each pixel coordinate is [v, u]
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float() # [2, H, W], each pixel coordinate is [u, v] after flip
        uv = uv.permute(1,2,0)[None] # [1, H, W, 2] 
        uv = uv.reshape(1,-1, 2)  # [1, H*W, 2]

        pose = self.pose_all[img_idx]  
        intrinsic = self.intrinsics_all[img_idx][None].cuda()

        rays_d, rays_o = rend_util.get_camera_params(uv.cuda(), pose[None].cuda(), intrinsic)
        rays_o = rays_o[:, None].expand_as(rays_d) 

        out = {'rays_o': rays_o[0],
               'rays_d': rays_d[0]}

        return out 

    def gen_random_rays_at(self, img_idx, batch_size, use_patch_uv=False, patch_num=0, r_patch=0):
        """
        Generate random rays at world space from one camera.
        """ 
        pose = self.pose_all[img_idx] 
        
        sample_index = torch.randperm(self.total_pixels)[:batch_size] 
        
        uv_full = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv_full = torch.from_numpy(np.flip(uv_full, axis=0).copy()) 
        uv_full = uv_full.reshape(2, -1).transpose(1, 0)
        uv = uv_full[None, sample_index, :].float() # 1, batch_size, 2  

        color = self.rgb_images[img_idx].reshape([-1,3])[sample_index, :].cuda()    # batch_size, 3
        mask = self.object_masks[img_idx].reshape(-1)[sample_index, None].float().cuda()   # batch_size, 1
        seg = self.object_segs[img_idx].reshape(-1)[sample_index, None].long().cuda()   # batch_size, 1
        normal = self.normal_images[img_idx].reshape([-1,3])[sample_index, :].cuda()    # batch_size, 3
        pretrain_normal = self.pretrain_normal_images[img_idx].reshape([-1,3])[sample_index, :].cuda() if self.use_pretrain_normal else None
        intrinsic = self.intrinsics_all[img_idx][None].cuda()
 
        pose = pose[None].cuda() # [batch_size,4,4] 
        
        rays_d, rays_o = rend_util.get_camera_params(uv.cuda(), pose, intrinsic)
        rays_o = rays_o[:, None].expand_as(rays_d)  

        out = {'rays_o': rays_o[0],
               'rays_d': rays_d[0],
               'color': color,
               'mask': mask,
               'seg': seg,
               'uv': uv,
               'normal': normal,
               'pretrain_normal': pretrain_normal,
               'pose': pose}

        if use_patch_uv:
            # r_patch: patch size will be (2*r_patch)*(2*r_patch)
            assert r_patch > 0 and patch_num > 0
 
            N_patch = patch_num 
            H, W = self.img_res
            u, v = np.meshgrid(np.arange(-r_patch, r_patch), np.arange(-r_patch, r_patch))
            
            u = u.reshape(-1)
            v = v.reshape(-1)
            offsets = v * W + u

            # center pixel coordinates
            mask = self.object_segs[img_idx][r_patch : W - r_patch,r_patch : H - r_patch]  
            mask = (mask > 0).bool()
            u, v = np.meshgrid(np.arange(r_patch, W - r_patch), np.arange(r_patch, H - r_patch)) 
            u = u[mask]
            v = v[mask] 
            select_inds = np.random.choice(u.shape[0], size=(N_patch,), replace=False)
            # convert back to original image
            select_inds = v[select_inds] * W + u[select_inds]
            # pick patches
            select_inds = np.stack([select_inds + shift for shift in offsets], axis=1)
            select_inds = select_inds.reshape(-1)
            sample_index = torch.from_numpy(select_inds).long()

            patch_uv = uv_full[None, sample_index, :].float() # 1, batch_size, 2

            patch_mask = self.object_masks[img_idx].reshape(-1)[sample_index, None].float().cuda()   # batch_size, 1
            patch_seg = self.object_segs[img_idx].reshape(-1)[sample_index, None].long().cuda()   # batch_size, 1
      
            patch_rays_d, patch_rays_o = rend_util.get_camera_params(patch_uv.cuda(), pose, intrinsic)
            patch_rays_o = patch_rays_o[:, None].expand_as(patch_rays_d)  

            out.update({
                'rays_o': torch.cat([out['rays_o'], patch_rays_o[0]], dim=0),
                'rays_d': torch.cat([out['rays_d'], patch_rays_d[0]], dim=0),  
                'mask': torch.cat([out['mask'], patch_mask], dim=0),
                'seg': torch.cat([out['seg'], patch_seg], dim=0),
                'uv': torch.cat([out['uv'], patch_uv], dim=1), 
            }) 
        return out    