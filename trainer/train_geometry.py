import os
import sys
import torch

import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from datetime import datetime
from tqdm.contrib import tzip
from termcolor import colored

import utils.general as utils
import utils.plots as plots
import utils.evaluate as evaluate

class GeometryTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = kwargs['conf']
        self.exps_folder_name = kwargs['exps_folder_name'] 
        self.end_iter = self.conf.train.neus_iter
        self.same_obj_num = kwargs['same_obj_num'] # how many objects in the scene
        self.visible_num = kwargs['visible_num']
        self.expname = 'Geo-' + kwargs['expname']
        self.select_index = kwargs['select_index']
        self.num_pixels = self.conf.train.geo_num_pixels
        self.debug = kwargs['debug']
        self.train_pose = self.conf.model.get('train_pose', False)
        self.real_world = kwargs['real_world']
        self.use_pretrain_normal = kwargs['use_pretrain_normal']
        self.progressive_training = self.conf.train.get('progressive_training', True)

        if self.use_pretrain_normal:
            print(colored('Using pretrained normal in training process', 'red', attrs=['bold']))
        if self.train_pose:
            print(colored('Training with pose optimization', 'red', attrs=['bold']))
        if self.progressive_training:
            print(colored('Use progressive training', 'red', attrs=['bold']))

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
        self.validate_resolution_level = self.conf.train.validate_resolution_level
        self.warm_up_end = self.conf.train.get('warm_up_end', 0.0) 

        self.is_continue = kwargs['is_continue']
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
        self.neus_optimizer_params_subdir = "NEUSOptimizerParameters" 

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.neus_optimizer_params_subdir)) 

        print('Write tensorboard to: ', os.path.join(self.expdir, self.timestamp))
        self.writer = SummaryWriter(os.path.join(self.expdir, self.timestamp))

        os.system("""cp -r {0} "{1}" """.format(self.conf.path, os.path.join(self.expdir, self.timestamp, 'setting.yaml')))
 
        print('Loading data ...')
        self.train_dataset = utils.get_class(self.conf.train.get('dataset_class'))(
            kwargs['data_split_dir'], kwargs['frame_skip'],
            select_index=self.select_index,
            split='train',
            same_obj_num=self.same_obj_num,
            visible_num=self.visible_num,
            non_empty_indexes=non_empty_indexes,
            real_world=kwargs['real_world'],
            use_pretrain_normal=self.use_pretrain_normal) 
        print('Finish loading data ...')

        self.model = utils.get_class(self.conf.train.get('model_class'))(
            same_obj_num=self.same_obj_num,
            visible_num=self.visible_num,
            scene_radius=self.train_dataset.scene_radius,
            scene_center=self.train_dataset.scene_center,
            conf=self.conf.model, 
            init_method=kwargs['init_method'],
            data_split_dir=kwargs['data_split_dir'],
            real_world=kwargs['real_world'],
            train_pose=self.train_pose,
            colmap_point=self.train_dataset.points_world_0,
        ) 
        if torch.cuda.is_available():
            self.model.cuda()
        
        self.loss = utils.get_class(self.conf.train.get('loss_class'))(use_pretrain_normal=self.use_pretrain_normal, **self.conf.loss)
        self.transformation_loss = utils.get_class(self.conf.train.transformation_loss_class)(**self.conf.transformation_loss) 
        
        neus_lr = self.conf.train.neus_learning_rate
        r_lr = self.conf.train.get('rotate_learning_rate', 1e-4)
        t_lr = self.conf.train.get('translate_learning_rate', 1e-4)
        print('Apply learning rate = {}, r = {}, t = {}'.format(neus_lr, r_lr, t_lr))
        if self.train_pose: 
            param_group = [ 
                {'params':self.model.rendering_network.parameters(), 'lr':neus_lr},
                {'params':self.model.deviation_network.parameters(), 'lr':neus_lr},
                {'params':self.model.point_classify_network.parameters(), 'lr':neus_lr},
                {'params':self.model.sdf_network.parameters(), 'lr':neus_lr},  
                {'params':self.model.obj_poses.object_q, 'lr':r_lr},
                {'params':self.model.obj_poses.object_t, 'lr':t_lr}
            ]
            if self.model.obj_poses.variant_vector_size > 0:
                param_group.append( {'params':self.model.obj_poses.variant_vector, 'lr':3e-5})
            self.neus_optimizer = torch.optim.AdamW(param_group)
        else:
            param_group = [    
                {'params':self.model.rendering_network.parameters(), 'lr':neus_lr},
                {'params':self.model.deviation_network.parameters(), 'lr':neus_lr},
                {'params':self.model.point_classify_network.parameters(), 'lr':neus_lr},
                {'params':self.model.sdf_network.parameters(), 'lr':neus_lr},
            ]
            if self.model.obj_poses.variant_vector_size > 0:
                param_group.append( {'params':self.model.obj_poses.variant_vector, 'lr':1e-4})
            self.neus_optimizer = torch.optim.AdamW(param_group)
        
        allow_part_load = True
        if is_continue and allow_part_load: 

            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            print('Loading pretrained model: ', os.path.join(
                old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))

            old_model_dict = self.model.state_dict()
            saved_model_state = torch.load(os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            saved_model_dict = saved_model_state["model_state_dict"]
 
            saved_model_dict = {k: v for k, v in saved_model_dict.items() if k in old_model_dict.keys()}
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
                os.path.join(old_checkpnts_dir, self.neus_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.neus_optimizer.load_state_dict(data["optimizer_state_dict"])
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataset)
        self.plot_freq = self.conf.train.geo_plot_freq
        self.ckpt_freq = self.conf.train.ckpt_freq

        # set initial active levels
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
            {"iter": cur_iter, "optimizer_state_dict": self.neus_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.neus_optimizer_params_subdir, str(cur_iter) + ".pth"))
        torch.save(
            {"iter": cur_iter, "optimizer_state_dict": self.neus_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.neus_optimizer_params_subdir, "latest.pth")) 
    
    def plot_to_disk(self, idx=-1, resolution_level=-1, eval_batch_size=2048):
        
        if idx < 0:
            idx = np.random.randint(len(self.train_dataset))
        
        self.model.eval()
        print('Validate: iter: {}, camera: {}'.format(self.cur_iter, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        
        ray_dict = self.train_dataset.gen_rays_at(idx)
        H, W = self.train_dataset.img_res

        rgb_gt = self.train_dataset.image_at(idx).cuda()
        mask = self.train_dataset.mask_at(idx).cuda()
        seg_gt = self.train_dataset.seg_at(idx).cuda()
        normal_gt = self.train_dataset.normal_at(idx).cuda()
        pretrain_normal_gt = self.train_dataset.pretrain_normal_at(idx)
        if pretrain_normal_gt is not None:
            pretrain_normal_gt = pretrain_normal_gt.cuda()

        rays_o = ray_dict['rays_o'].split(eval_batch_size)
        rays_d = ray_dict['rays_d'].split(eval_batch_size)  
        rays_seg = seg_gt.reshape(-1, 1).split(eval_batch_size)  

        out_rgb_fine = []
        out_normal_fine = []
        out_network_mask = []
        out_proxy_pred_class = []  
        
        for rays_o_batch, rays_d_batch, rays_seg_batch in tzip(rays_o, rays_d, rays_seg):
            near, far = self.train_dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

            render_out = self.model(rays_o_batch.cuda(),
                                    rays_d_batch.cuda(),
                                    near.cuda(),
                                    far.cuda(),
                                    rays_seg_batch,
                                    cos_anneal_ratio=self.get_cos_anneal_ratio(), 
                                    perturb_overwrite=0,
                                    trainstage='Geo',
                                    is_eval=True)

            out_rgb_fine.append(render_out['color'].detach()) 
            out_normal_fine.append(render_out['surface_normal'].detach()) 
            out_network_mask.append(render_out['network_object_mask'].detach())
            out_proxy_pred_class.append(render_out['surface_proxy_pred_class'].detach()) 
            del render_out 

        rgb_img = torch.cat(out_rgb_fine, dim=0).reshape([H, W, 3]) 

        normal_img_world = torch.cat(out_normal_fine, dim=0).reshape([-1, 3]) 
        rot = torch.linalg.inv(self.train_dataset.pose_all[idx][:3, :3]).cuda() # world to camera
        normal_img_cam = torch.matmul(rot[None, :, :], normal_img_world[:, :, None]).reshape([H, W, 3])

        network_mask = torch.cat(out_network_mask, dim=0).reshape([H, W])
         
        proxy_object_seg = torch.cat(out_proxy_pred_class, dim=0).reshape([1, H*W, -1]) 
        
        network_mask = (network_mask>self.model.weight_threshold)

        plots.plot_neus(
            rgb_pred=rgb_img,
            normal_img=normal_img_cam,
            rgb_gt=rgb_gt, 
            normal_gt=normal_gt,
            object_mask=mask,
            network_mask=network_mask,
            path=self.plots_dir,
            iters=self.cur_iter,
            img_res=self.img_res,
            gt_seg=seg_gt, 
            sdf_pred_seg=None,
            proxy_pred_seg=proxy_object_seg,
            same_obj_num=self.visible_num,
            pretrain_normal_gt=pretrain_normal_gt,
            use_pretrain_normal=self.use_pretrain_normal,
        )

        evaluate.evaluate_all(
            pred_rgb=rgb_img,
            object_mask=mask,
            network_mask=network_mask,
            rgb_gt=rgb_gt,
            normal_gt=None,
            path=self.plots_dir,
            iters=self.cur_iter,
            img_res=self.img_res,
            use_union_mask=True,
            )
        evaluate.evaluate_all(
            pred_rgb=rgb_img,
            object_mask=mask,
            network_mask=network_mask,
            rgb_gt=rgb_gt,
            normal_gt=None,
            path=self.plots_dir,
            iters=self.cur_iter,
            img_res=self.img_res,
            use_union_mask=False,)

        self.model.obj_poses.get_diff(image_path=os.path.join(self.plots_dir, 'pose_error_{}.png'.format(self.cur_iter))) 
        print('save pose error to {}'.format(os.path.join(self.plots_dir, 'pose_error_{}.png'.format(self.cur_iter))))
        self.model.train()
    
    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.cur_iter / self.anneal_end])

    def run(self):
        
        print("training...")  

        image_perm = torch.randperm(len(self.train_dataset)) 

        # import pdb
        # pdb.set_trace()
 
        while(self.cur_iter <= self.end_iter):  
            
            progress = 1
            if self.progressive_training and self.conf.model.use_hash:
                progress = np.clip((self.cur_iter - 2500) / 10000, 0, 1)
                self.model.sdf_network.set_active_levels(current_iter=self.cur_iter, warm_up_end=5000)
                self.model.sdf_network.set_normal_epsilon()
            elif self.progressive_training and not self.conf.model.use_hash:
                progress = np.clip((self.cur_iter - 20000) / 30000, 0, 1)
                
            self.model.sdf_network.progress = progress
            self.model.rendering_network.progress = progress
            self.model.point_classify_network.progress = progress

            if self.cur_iter > self.max_iter:
                self.save_checkpoints(self.cur_iter)
                self.plot_to_disk()
                print('Training has reached max number of iterations: {}; exiting...'.format(self.cur_iter))
                exit(0)
            
            if self.cur_iter % self.ckpt_freq == 0:
                self.save_checkpoints(self.cur_iter) 

            if self.cur_iter % self.plot_freq == 0 and not self.debug :
                self.plot_to_disk()

            ray_dict = self.train_dataset.gen_random_rays_at(image_perm[self.cur_iter % len(image_perm)], self.num_pixels) 

            rays_o = ray_dict['rays_o']
            rays_d = ray_dict['rays_d'] 
            mask = ray_dict['mask']
            object_seg = ray_dict['seg']
            near, far = self.train_dataset.near_far_from_sphere(rays_o, rays_d)

            render_out = self.model(rays_o, rays_d, near, far, object_seg, cos_anneal_ratio=self.get_cos_anneal_ratio(), trainstage='Geo')
            loss_output = self.loss(model_output=render_out, ground_truth=ray_dict, real_world=self.real_world)

            neus_total_loss = loss_output['neus_total_loss'] 
            # ground_truth = {'object_seg': object_seg.transpose(1,0)}
            render_out.update({'object_mask': mask[:,0]}) 
            
            transformation_loss = self.transformation_loss(render_out)
            transformation_total_loss = transformation_loss['transformation_total_loss'] 

            vec_var = self.model.obj_poses.get_var_of_variant_vector()
            var_loss = vec_var * 0.05
            total_loss = neus_total_loss + var_loss
            self.neus_optimizer.zero_grad() 
            (total_loss+transformation_total_loss).backward()
            self.neus_optimizer.step()

            if self.cur_iter % 5 == 0:   
                dr, dt = self.model.obj_poses.get_diff() 
                print('iter [{}]: loss = {}, s_val = {:.6f} progress = {:.4f}\n'
                    'rgb = {:.4f}, eik = {:.4f}, hes = {:.4f}, mask = {:.4f}, colmap = {:.4f}, normal = {:.4f}\n'
                    'self = {:.5f}, seg = {:.6f}, dr = {:.5f}, dt = {:.5f}\n'
                    'density = {:.5f}, var loss = {:.10f}, var = {:.10f}\n'
                        .format(self.cur_iter,
                                total_loss.item(),
                                render_out['s_val'][0].item(), 
                                progress,
                                
                                loss_output['color_loss'].item(),
                                loss_output['eikonal_loss'].item(),
                                loss_output['hessian_loss'].item(),
                                loss_output['mask_loss'].item(),
                                loss_output['colmap_sdf_loss'].item(),
                                loss_output['normal_loss'].item(),  

                                transformation_loss['distillation_loss'].item(),
                                transformation_loss['segmentation_loss'].item(),   
                                # pose_loss.item(),
                                dr.mean().item(),
                                dt.mean().item(),
                                render_out['weights_sum'].mean().item() * 10,
                                var_loss.item(),
                                vec_var.item(),

                                # self.neus_scheduler.get_lr() if self.train_pose else self.neus_optimizer.param_groups[0]['lr'],
                                ))
                self.writer.add_scalar('total_loss', total_loss.item(), self.cur_iter)
                self.writer.add_scalar('rgb_loss', loss_output['color_loss'].item(), self.cur_iter) 
                self.writer.add_scalar('eikonal_loss', loss_output['eikonal_loss'].item(), self.cur_iter)
                self.writer.add_scalar('mask_loss', loss_output['mask_loss'].item(), self.cur_iter) 
                self.writer.add_scalar('normal_loss', loss_output['normal_loss'].item(), self.cur_iter)

                self.writer.add_scalar('distillation_loss', transformation_loss['distillation_loss'].item(), self.cur_iter)
                self.writer.add_scalar('segmentation_loss', transformation_loss['segmentation_loss'].item(), self.cur_iter)

                self.writer.add_scalar('s_val', render_out['s_val'][0].item(), self.cur_iter)

                self.writer.add_scalar('dr_mean', dr.mean().item(), self.cur_iter)
                self.writer.add_scalar('dt_mean', dt.mean().item(), self.cur_iter)
                self.writer.add_scalar('vec_var', vec_var.item(), self.cur_iter)
                for i in range(len(dr)): 
                    self.writer.add_scalar('dr_{}'.format(i), dr[i].item(), self.cur_iter)
                    self.writer.add_scalar('dt_{}'.format(i), dt[i].item(), self.cur_iter)
                
            if self.cur_iter % 50 == 0 and self.train_pose and not self.real_world:
                dr, dt = self.model.obj_poses.get_diff()
                for j in range(dr.shape[0]):
                    write_path = os.path.join(os.path.dirname(self.plots_dir), 'diff_pose_{0}.txt'.format(j))
                    with open(write_path, "a+")as f:
                        f.write('iters [{0}]: \n'.format(self.cur_iter))
                        f.write('dr = {}, dt = {} \n'.format(str(dr[j]), str(dt[j]))) 

                write_path = os.path.join(os.path.dirname(self.plots_dir), 'diff_pose_mean.txt')
                with open(write_path, "a+")as f:
                    f.write('iters [{0}]: \n'.format(self.cur_iter))
                    f.write('dr = {}, dt = {} \n'.format(str(dr.mean()), str(dt.mean()))) 

            if self.cur_iter % len(image_perm) == 0:
                image_perm = torch.randperm(len(self.train_dataset))

            self.cur_iter += 1 