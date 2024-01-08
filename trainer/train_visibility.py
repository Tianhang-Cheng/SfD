import os
import sys
from datetime import datetime
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import numpy as np
from tqdm.contrib import tzip
import utils.general as utils
import utils.plots as plots 

class VisbilityTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = kwargs['conf']
        self.exps_folder_name = kwargs['exps_folder_name'] 
        self.end_iter = self.conf.train.illum_iter
        self.single_image = kwargs['single_image']
 
        self.expname = 'Vis-' + kwargs['expname']
        self.same_obj_num = kwargs['same_obj_num']
        self.visible_num = kwargs['visible_num']
        self.select_index = kwargs['select_index']
        self.num_pixels = self.conf.train.vis_num_pixels
        self.real_world = kwargs['real_world']
        self.debug = kwargs['debug']
        self.train_pose = kwargs['train_pose']

        non_empty_path = os.path.join(kwargs['data_split_dir'], 'non_empty_indexes.txt')
        assert os.path.exists(non_empty_path)
        non_empty_indexes = np.loadtxt(non_empty_path).astype(int)

        assert self.visible_num == -1 or self.visible_num <= self.same_obj_num, 'visible num should less than total num'
        assert self.visible_num <= len(non_empty_indexes), 'visible num should less than num of good instances'
        if self.visible_num == -1:
            self.visible_num = len(non_empty_indexes)

        if kwargs['is_eval']:
            raise NotImplementedError
        
        # training parameter
        self.cur_iter = 0
        self.max_iter = kwargs['max_niters']
        self.anneal_end = self.conf.train.get('anneal_end', 0.0)
        self.use_white_bkgd = False
        self.validate_resolution_level = self.conf.get('validate_resolution_level')
        self.warm_up_end = self.conf.train.get('warm_up_end', 0.0)
        self.learning_rate_alpha = self.conf.train.get('learning_rate_alpha')
        
        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join(kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join(kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        utils.mkdir_ifnotexists(self.exps_folder_name)
        self.expdir = os.path.join(self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.neus_vis_optimizer_params_subdir = "NEUSVisOptimizerParameters" 

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.neus_vis_optimizer_params_subdir)) 

        print('Write tensorboard to: ', os.path.join(self.expdir, self.timestamp))
        self.writer = SummaryWriter(os.path.join(self.expdir, self.timestamp))
 
        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'setting.yaml')))
 
        print('Loading data ...')
        self.train_dataset = utils.get_class(self.conf.train.get('dataset_class'))(
            kwargs['data_split_dir'], kwargs['frame_skip'], 
            select_index=self.select_index,
            visible_num=self.visible_num,
            same_obj_num=self.same_obj_num,
            non_empty_indexes=non_empty_indexes,
            split='train',
            real_world=self.real_world) 
       
        print('Finish loading data ...')
 
        self.model = utils.get_class(self.conf.train.get('model_class'))( 
            same_obj_num=self.same_obj_num,
            visible_num=self.visible_num,
            init_method=kwargs['init_method'],
            data_split_dir=kwargs['data_split_dir'],
            conf=self.conf.model,
            real_world=self.real_world,
            train_pose=self.train_pose,
            scene_radius=self.train_dataset.scene_radius,
            scene_center=self.train_dataset.scene_center,
            use_colmap_constraint=False,
        )
        if torch.cuda.is_available():
            self.model.cuda()
        
        self.learning_rate = self.conf.train.get('illum_learning_rate')
        self.loss = utils.get_class(self.conf.train.get('illum_loss_class'))()
        self.neus_vis_optimizer = torch.optim.Adam(list(self.model.visibility_network.parameters()), lr=self.learning_rate) 

        allow_part_load = True
        if is_continue and allow_part_load:

            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            print('Loading pretrained model: ', os.path.join(
                old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))

            old_model_dict = self.model.state_dict()
            saved_model_state = torch.load(os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            saved_model_dict = saved_model_state["model_state_dict"]
            saved_model_dict = {k: v for k, v in saved_model_dict.items() if k in old_model_dict}
            old_model_dict.update(saved_model_dict)
            self.model.load_state_dict(old_model_dict)   
            self.cur_iter = saved_model_state['iter']

        elif is_continue and not allow_part_load:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            print('Loading pretrained model: ', os.path.join(
                old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])   
            self.cur_iter = saved_model_state['iter']

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.neus_vis_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.neus_vis_optimizer.load_state_dict(data["optimizer_state_dict"])

        else:

            geo_dir = os.path.join(kwargs['exps_folder_name'], 'Geo-' + kwargs['expname'])
            if os.path.exists(geo_dir):
                timestamps = os.listdir(geo_dir)
                timestamp = sorted(timestamps)[-1] # using the newest training result
            else:
                print('No geometry pretrain, please train neus first!')
                exit(0)

            # reloading geometry
            geo_path = os.path.join(geo_dir, timestamp) + '/checkpoints/ModelParameters/latest.pth'
            print('Reloading geometry from: ', geo_path)
            old_dict = torch.load(geo_path)['model_state_dict']
            old_dict = {k:old_dict[k] for k in old_dict.keys() if 'deform' not in k}
            new_dict = self.model.state_dict()
            new_dict.update(old_dict)
            self.model.load_state_dict(new_dict) 

        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataset)
        self.plot_freq = self.conf.train.vis_plot_freq
        self.ckpt_freq = self.conf.train.ckpt_freq

        if self.conf.model.use_hash:
            self.model.sdf_network.set_active_levels(current_iter=self.conf.train.neus_iter, warm_up_end=5000)
            self.model.sdf_network.set_normal_epsilon()
    
    def save_checkpoints(self, cur_iter):
        torch.save(
            {"iter": cur_iter, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(cur_iter) + ".pth"))
        torch.save(
            {"iter": cur_iter, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"iter": cur_iter, "optimizer_state_dict": self.neus_vis_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.neus_vis_optimizer_params_subdir, str(cur_iter) + ".pth"))
        torch.save(
            {"iter": cur_iter, "optimizer_state_dict": self.neus_vis_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.neus_vis_optimizer_params_subdir, "latest.pth"))
  
    
    def plot_to_disk(self, idx=-1, resolution_level=-1, eval_batch_size=2048):
        
        if idx < 0:
            idx = np.random.randint(len(self.train_dataset))
        
        self.model.eval()
        print('Validate: iter: {}, camera: {}'.format(self.cur_iter, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        
        seg_gt = self.train_dataset.seg_at(idx).cuda()

        ray_dict = self.train_dataset.gen_rays_at(idx)
        H, W = self.train_dataset.img_res
        rays_o = ray_dict['rays_o'].split(eval_batch_size)
        rays_d = ray_dict['rays_d'].split(eval_batch_size) 
        rays_seg = seg_gt.reshape(-1, 1).split(eval_batch_size)  

        pred_vis = []
        gt_vis = []  
        network_object_mask = []
        
        for rays_o_batch, rays_d_batch, rays_seg_batch in tzip(rays_o, rays_d, rays_seg): 
            a = torch.cuda.memory_allocated()
            b = torch.cuda.max_memory_allocated() 
            print('{}/{} = {}'.format(a,b,a/b))    
            near, far = self.train_dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

            render_out = self.model.trace_visibility(rays_o_batch.cuda(),
                                                    rays_d_batch.cuda(),
                                                    near.cuda(),
                                                    far.cuda(),
                                                    rays_seg_batch,
                                                    cos_anneal_ratio=self.get_cos_anneal_ratio())

            pred_vis.append(render_out['pred_vis'].detach())
            gt_vis.append(render_out['gt_vis'].detach()) 
            network_object_mask.append(render_out['network_object_mask'].detach())
            torch.cuda.empty_cache()

            del render_out 
 
        if len(pred_vis) > 0:
            pred_vis = torch.cat(pred_vis, dim=0) 
 
        if len(gt_vis) > 0:
            gt_vis = torch.cat(gt_vis, dim=0) 
        
        if len(network_object_mask) > 0:
            network_object_mask = torch.cat(network_object_mask, dim=0) 
         
        mask = self.train_dataset.mask_at(idx).cuda().flatten()
        network_object_mask = network_object_mask.bool().flatten()
        mask = mask & network_object_mask

        pred_vis = F.softmax(pred_vis.detach(), dim=-1)[..., 1]
        # pred_vis = torch.max(pred_vis.detach(), dim=-1)[1].float()
        pred_vis = torch.mean(pred_vis.float(), axis=1)
        
        pred_vis[~mask] = 1.0
        gt_vis = torch.mean((~gt_vis).float(), axis=1)[:, 0] 
        gt_vis[~mask] = 1.0
        
        plots.plot_neus_vis(
            pred_vis=pred_vis.reshape([H,W]),
            gt_vis=gt_vis.reshape([H,W]),  
            rgb_gt=self.train_dataset.image_at(idx).cuda() if not self.single_image else None,
            path=self.plots_dir,
            iters=self.cur_iter,
            img_res=self.img_res
        )
         
        self.model.train() 
    
    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def run(self):
        print("training...")  

        image_perm = torch.randperm(len(self.train_dataset))
 
        while(self.cur_iter <= self.end_iter): 

            if self.cur_iter > self.max_iter or self.cur_iter >= 3000:
                self.save_checkpoints(self.cur_iter)
                self.plot_to_disk()
                print('Training has reached max number of iterations: {}; exiting...'.format(self.cur_iter))
                exit(0)
            
            if self.cur_iter % self.ckpt_freq == 0:
                self.save_checkpoints(self.cur_iter) 

            # if self.cur_iter % self.plot_freq == 0 and not self.debug:
            #     self.plot_to_disk() 
 
            ray_dict = self.train_dataset.gen_random_rays_at(image_perm[self.cur_iter % len(image_perm)], self.num_pixels) 
            rays_o = ray_dict['rays_o']
            rays_d = ray_dict['rays_d']  
            near, far = self.train_dataset.near_far_from_sphere(rays_o, rays_d)
            object_seg = ray_dict['seg'] 
            
            render_out = self.model.trace_visibility(rays_o, rays_d, near, far, object_seg, cos_anneal_ratio=self.get_cos_anneal_ratio()) 

            loss_output = self.loss(model_output=render_out)
            loss = loss_output['visibility_loss']
            self.neus_vis_optimizer.zero_grad()
            loss.backward()
            self.neus_vis_optimizer.step() 

            if self.cur_iter % 10 == 0:
                print('{0} [{1}]: loss = {2}, lr = {3}'.format(self.expname, self.cur_iter, loss.item(), self.neus_vis_optimizer.param_groups[0]['lr']))

            if self.cur_iter % len(image_perm) == 0:
                image_perm = torch.randperm(len(self.train_dataset))
            
            self.cur_iter += 1 