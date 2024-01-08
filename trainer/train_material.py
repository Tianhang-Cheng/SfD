import os
import sys
import time
import torch
import imageio

import numpy as np
from tqdm import tqdm
from datetime import datetime
from tqdm.contrib import tzip
from tensorboardX import SummaryWriter

from model.material_sg import compute_envmap
from utils import rend_util
import utils.general as utils
import utils.plots as plots
import utils.evaluate as evaluate

tonemap_img = lambda x: torch.pow(x, 1./2.2) 
clip_img = lambda x: torch.clamp(x, min=0., max=1.) 

class MaterialTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = kwargs['conf']
        self.exps_folder_name = kwargs['exps_folder_name'] 
        self.end_iter = self.conf.train.sg_iter

        self.same_obj_num = kwargs['same_obj_num']
        self.visible_num = kwargs['visible_num']
        self.input_name = kwargs['expname']
        self.expname = 'Mat-' + kwargs['expname']
        self.select_index = kwargs['select_index']
        self.is_eval = kwargs['is_eval']
        self.is_eval_relight = kwargs['is_eval_relight'] 
        self.num_pixels = self.conf.train.mat_num_pixels
        self.real_world = kwargs['real_world']
        self.use_pretrain_normal = kwargs['use_pretrain_normal']
        self.to_uv = kwargs['to_uv']
        self.to_mesh = kwargs['to_mesh']
        self.debug = kwargs['debug'] 
        self.train_pose = kwargs['train_pose']

        non_empty_path = os.path.join(kwargs['data_split_dir'], 'non_empty_indexes.txt')
        assert os.path.exists(non_empty_path)
        non_empty_indexes = np.loadtxt(non_empty_path).astype(int)

        assert self.visible_num == -1 or self.visible_num <= self.same_obj_num, 'visible num should less than total num'
        assert self.visible_num <= len(non_empty_indexes), 'visible num should less than num of good instances'
        if self.visible_num == -1:
            self.visible_num = len(non_empty_indexes)

        self.name = 'our' 

        # training parameter
        self.cur_iter = 0
        self.max_iter = kwargs['max_niters'] 
        self.use_white_bkgd = False
        self.validate_resolution_level = self.conf.train.validate_resolution_level

        if self.is_eval_relight:
            self.is_eval = True
 
        if (kwargs['is_continue'] or self.is_eval or self.to_uv or self.to_mesh) and kwargs['timestamp'] == 'latest':
            
            if os.path.exists(os.path.join( kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join( kwargs['exps_folder_name'],self.expname))
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
        
        self.old_expname = self.expname
        if self.is_eval_relight:
            self.expname = self.expname + '-eval-relight'
        elif self.is_eval:
            self.expname = self.expname + '-eval'
        elif self.to_mesh:
            self.expname = self.expname + '-mesh'
        
        self.expdir = os.path.join(self.exps_folder_name, self.expname)
        self.old_expdir = os.path.join(self.exps_folder_name, self.old_expname)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        self.model_params_subdir = "ModelParameters"
        self.neus_mat_optimizer_params_subdir = "NEUSMatOptimizerParameters" 
        self.eval_dataset = None
        self.train_dataset = None

        if not self.to_uv:
            
            utils.mkdir_ifnotexists(self.exps_folder_name)
            utils.mkdir_ifnotexists(self.expdir)    
            utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

            if not self.is_eval:
                self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
                utils.mkdir_ifnotexists(self.plots_dir)

                # create checkpoints dirs
                self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
                utils.mkdir_ifnotexists(self.checkpoints_path)


                utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
                utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.neus_mat_optimizer_params_subdir)) 

                print('Write tensorboard to: ', os.path.join(self.expdir, self.timestamp))
                self.writer = SummaryWriter(os.path.join(self.expdir, self.timestamp))
            elif not self.is_eval_relight:
                self.eval_dir_image = os.path.join(self.expdir, self.timestamp, 'evals_image')
                self.eval_dir_value = os.path.join(self.expdir, self.timestamp, 'evals_value')
                utils.mkdir_ifnotexists(self.eval_dir_image)
                utils.mkdir_ifnotexists(self.eval_dir_value)  
 
        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'setting.yaml')))

        print('Loading data ...') 

        if self.is_eval:
            split = 'test_relight' if self.is_eval_relight else 'test'
            self.eval_dataset = utils.get_class(self.conf.train.dataset_class)(
                kwargs['data_split_dir'], kwargs['eval_frame_skip'],
                select_index=self.select_index,
                split=split,
                same_obj_num=self.same_obj_num,
                visible_num=self.visible_num,
                non_empty_indexes=non_empty_indexes, 
                real_world=self.real_world,
                use_pretrain_normal=self.use_pretrain_normal) 
        else:
            self.train_dataset = utils.get_class(self.conf.train.dataset_class)(
                kwargs['data_split_dir'], kwargs['frame_skip'], 
                select_index=self.select_index,
                split='train',
                same_obj_num=self.same_obj_num,
                visible_num=self.visible_num,
                non_empty_indexes=non_empty_indexes, 
                real_world=self.real_world,
                use_pretrain_normal=self.use_pretrain_normal)  
        print('Finish loading data ...')

        scene_radius = self.eval_dataset.scene_radius if self.is_eval else self.train_dataset.scene_radius
        scene_center = self.eval_dataset.scene_center if self.is_eval else self.train_dataset.scene_center

        self.model = utils.get_class(self.conf.train.model_class)(
            same_obj_num=self.same_obj_num,
            visible_num=self.visible_num, 
            data_split_dir=kwargs['data_split_dir'],
            init_method=kwargs['init_method'],
            conf=self.conf.model, 
            real_world=self.real_world,
            train_pose=self.train_pose,
            scene_radius=scene_radius,
            scene_center=scene_center,
            use_colmap_constraint=False,
        )

        if torch.cuda.is_available():
            self.model.cuda()
        
        if self.train_dataset is None:
            self.scene_center = self.eval_dataset.scene_center.cuda()
        else:
            self.scene_center = self.train_dataset.scene_center.cuda()
    
        self.learning_rate = self.conf.train.sg_learning_rate
        self.loss = utils.get_class(self.conf.train.mat_loss_class)(brdf_multires=self.conf.model.envmap_material_network.multires, **self.conf.mat_loss)
        self.transformation_loss = utils.get_class(self.conf.train.transformation_loss_class)(**self.conf.transformation_loss)
        self.geo_loss = utils.get_class(self.conf.train.loss_class)(use_pretrain_normal=self.use_pretrain_normal, **self.conf.loss)

        self.neus_mat_optimizer = torch.optim.Adam([
                {'params':self.model.envmap_material_network.parameters(), 'lr':self.learning_rate},
                # {'params':self.model.deviation_network.parameters(), 'lr':self.learning_rate}, 
                # {'params':self.model.sdf_network.parameters(), 'lr': self.learning_rate / 3}, 
                # {'params':self.model.obj_poses.parameters(), 'lr':1e-5},
        ])

        # reloading
        if not is_continue:
            vis_dir = os.path.join(kwargs['exps_folder_name'], 'Vis-' + kwargs['expname'])
            if os.path.exists(vis_dir):
                timestamps = os.listdir(vis_dir)
                vis_timestamp = sorted(timestamps)[-1] # using the newest training result
            else:
                print('No visbility pretrain, please train neus first!')
                exit(0)
            vis_path = os.path.join(vis_dir, vis_timestamp) + '/checkpoints/ModelParameters/latest.pth'
            print('Reloading visibility from: ', vis_path)  
            old_model_dict = self.model.state_dict()
            vis_model_dict = {k : v for k, v in torch.load(vis_path)['model_state_dict'].items() if k in old_model_dict.keys() } 
            old_model_dict.update(vis_model_dict)
            self.model.load_state_dict(old_model_dict) 
 
        if is_continue or self.is_eval or self.to_uv or self.to_mesh:
            old_checkpnts_dir = os.path.join(self.old_expdir, timestamp, 'checkpoints')

            print('Loading pretrained model: ', os.path.join(
                old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            
            # print(self.model.envmap_material_network.brdf_encoder_layer[0].weight) 
            old_model_dict = self.model.state_dict()
            saved_model_state = torch.load(os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth")) 
            saved_model_dict = saved_model_state["model_state_dict"]
            saved_model_dict = {k: v for k, v in saved_model_dict.items() if k in old_model_dict.keys()} 
            old_model_dict.update(saved_model_dict)
            # self.model.envmap_material_network.load_state_dict(saved_model_dict)   
            self.model.load_state_dict(old_model_dict)   
            # self.model.envmap_material_network.load_state_dict(saved_model_dict)   
            self.cur_iter = saved_model_state['iter'] 

            # print(self.model.envmap_material_network.brdf_encoder_layer[0].weight)

            # data = torch.load(os.path.join(old_checkpnts_dir, self.neus_mat_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            # self.neus_mat_optimizer.load_state_dict(data["optimizer_state_dict"])
    
        if not self.to_uv and not self.to_mesh:
            self.img_res = self.eval_dataset.img_res if self.is_eval else self.train_dataset.img_res 
            self.plot_freq = self.conf.train.mat_plot_freq
            self.ckpt_freq = self.conf.train.ckpt_freq

            exr_path = os.path.join(sys.path[0], 'envmaps', 'c.exr')
            print('GT env light path is {}'.format(exr_path))
            gt_envmap = torch.from_numpy(rend_util.load_exr(exr_path)).cuda().permute(2,0,1)[None] # [1,3,h,w]
            gt_envmap = torch.nn.functional.interpolate(gt_envmap, size=(256,512), mode='bilinear')[0].permute(1,2,0) # [h,w,3]
            self.gt_envmap = tonemap_img(clip_img(gt_envmap)).cpu().numpy().astype(np.float32)

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
            {"iter": cur_iter, "optimizer_state_dict": self.neus_mat_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.neus_mat_optimizer_params_subdir, str(cur_iter) + ".pth"))
        torch.save(
            {"iter": cur_iter, "optimizer_state_dict": self.neus_mat_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.neus_mat_optimizer_params_subdir, "latest.pth"))

    def plot_to_disk(self, idx=-1, resolution_level=-1, eval_batch_size=2048):
        
        if idx < 0:
            idx = np.random.randint(len(self.train_dataset))
        
        self.model.eval()
        print('Validate: iter: {}, camera: {}'.format(self.cur_iter, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        
        ray_dict = self.train_dataset.gen_rays_at(idx)
        H, W = self.train_dataset.img_res
        rays_o = ray_dict['rays_o'].split(eval_batch_size)
        rays_d = ray_dict['rays_d'].split(eval_batch_size) 
        seg_gt = self.train_dataset.seg_at(idx).cuda()
        rays_seg = seg_gt.reshape(-1, 1).split(eval_batch_size) 

        sg_rgb = []
        specular_rgb = []
        normals = []  
        roughness = []
        diffuse_albedo = []
        metallic = []
        vis_shadow = []
        network_object_mask = []
        network_object_mask_draw = []
        points = []

        for rays_o_batch, rays_d_batch, rays_seg_batch in tzip(rays_o, rays_d, rays_seg): 

            near, far = self.train_dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

            render_out = self.model(rays_o_batch.cuda(),
                                    rays_d_batch.cuda(),
                                    near.cuda(),
                                    far.cuda(), 
                                    rays_seg_batch,
                                    cos_anneal_ratio=self.get_cos_anneal_ratio(), 
                                    trainstage='Mat',
                                    is_eval=True)

            sg_rgb.append(render_out['sg_rgb'].detach())
            specular_rgb.append(render_out['sg_specular_rgb'].detach()) 
            normals.append(render_out['normals'].detach()) 
            roughness.append(render_out['roughness'].detach())
            metallic.append(render_out['metallic'].detach())
            diffuse_albedo.append(render_out['diffuse_albedo'].detach())
            vis_shadow.append(render_out['vis_shadow'].detach())
            network_object_mask.append(render_out['network_object_mask'].detach())
            network_object_mask_draw.append(render_out['network_object_mask_draw'].detach())
            points.append(render_out['points'].detach())

            del render_out 
 
        if len(sg_rgb) > 0:
            sg_rgb = torch.cat(sg_rgb, dim=0) 
        if len(specular_rgb) > 0:
            specular_rgb = torch.cat(specular_rgb, dim=0)
        if len(normals) > 0:
            normals = torch.cat(normals, dim=0) 
        if len(roughness) > 0:
            roughness = torch.cat(roughness, dim=0) 
        if len(metallic) > 0:
            metallic = torch.cat(metallic, dim=0) 
        if len(diffuse_albedo) > 0:
            diffuse_albedo = torch.cat(diffuse_albedo, dim=0) 
        if len(vis_shadow) > 0:
            vis_shadow = torch.cat(vis_shadow, dim=0) 
        if len(network_object_mask) > 0:
            network_object_mask = torch.cat(network_object_mask, dim=0)
        if len(network_object_mask_draw) > 0:
            network_object_mask_draw = torch.cat(network_object_mask_draw, dim=0)
        if len(points) > 0:
            points = torch.cat(points, dim=0)
         
        object_mask = self.train_dataset.mask_at(idx).cuda()
        rgb_gt = self.train_dataset.image_at(idx).cuda()  
        # depth_gt = self.train_dataset.depth_at(idx).cuda()
        pose = self.train_dataset.pose_at(idx).cuda()
        normals_gt = self.train_dataset.normal_at(idx).cuda()
 
        plots.plot_neus_mat(
            normal=normals.reshape([H,W,3]), 
            normal_gt=normals_gt.reshape([H,W,3]),
            vis_shadow=vis_shadow.reshape([H,W,3]),
            diffuse_albedo=diffuse_albedo.reshape([H,W,3]), 
            diffuse_albedo_gt=None, 
            pose=pose,
            # depth_gt=depth_gt,
            points=points.reshape([1,H*W,3]),
            roughness=roughness.reshape([H,W,3]),
            roughness_gt=None,
            metallic=metallic.reshape([H,W,3]),
            metallic_gt=None,
            specular_rgb=specular_rgb.reshape([H,W,3]),
            sg_rgb=sg_rgb.reshape([H,W,3]),
            rgb_gt=rgb_gt.reshape([H,W,3]),
            obj_mask=object_mask.reshape([H,W,1]),
            network_obj_mask=network_object_mask_draw.reshape([H,W,1]),
            path=self.plots_dir,
            iters=self.cur_iter,
            sep=True,
            )

        evaluate.evaluate_all(
            pred_rgb=sg_rgb.reshape(-1,3),
            object_mask=object_mask.flatten(),
            network_mask=network_object_mask.flatten(),
            rgb_gt=rgb_gt.reshape(-1,3),
            points=points.reshape(-1,3),
            # depth_gt=depth_gt.flatten(),
            normals=normals.reshape(-1,3),
            normals_gt=normals_gt.reshape(-1,3),
            roughness=roughness.reshape(-1,3),
            roughness_gt=None,
            pose=pose.reshape(1,4,4), 
            path=self.plots_dir,
            iters=self.cur_iter,
            img_res=self.img_res,
            use_union_mask=True,
            prefix='rgb')
        evaluate.evaluate_all(
            pred_rgb=sg_rgb.reshape(-1,3),
            object_mask=object_mask.flatten(),
            network_mask=network_object_mask.flatten(),
            rgb_gt=rgb_gt.reshape(-1,3),
            points=points.reshape(-1,3),
            # depth_gt=depth_gt.flatten(),
            normals=normals.reshape(-1,3),
            normals_gt=normals_gt.reshape(-1,3),
            roughness=roughness.reshape(-1,3),
            roughness_gt=None,
            pose=pose.reshape(1,4,4), 
            path=self.plots_dir,
            iters=self.cur_iter,
            img_res=self.img_res,
            use_union_mask=False,
            prefix='rgb')

        # log environment map
        lgtSGs = self.model.envmap_material_network.get_light()
        envmap = compute_envmap(
            lgtSGs=lgtSGs, H=256, W=512, upper_hemi=self.model.envmap_material_network.upper_hemi)
        envmap = envmap.cpu().numpy()
        imageio.imwrite(os.path.join(self.plots_dir, 'envmap1_{}.exr'.format(self.cur_iter)), envmap)

        self.model.train() 
    
    def get_cos_anneal_ratio(self):
        return 1.0
        
    def run(self):
        print("training...") 
        mse2psnr = lambda x: -10. * np.log(x + 1e-8) / np.log(10.)

        image_perm = torch.randperm(len(self.train_dataset))
 
        while(self.cur_iter <= self.end_iter): 
 
            progress = 1.0
            self.model.envmap_material_network.progress = progress

            if self.cur_iter > self.max_iter:
                self.save_checkpoints(self.cur_iter)
                self.plot_to_disk()
                print('Training has reached max number of iterations: {}; exiting...'.format(self.cur_iter))
                exit(0)
            
            if self.cur_iter % self.ckpt_freq == 0:
                self.save_checkpoints(self.cur_iter)

            if self.cur_iter % self.plot_freq == 0 and not self.debug:
                self.plot_to_disk()
 
            ray_dict = self.train_dataset.gen_random_rays_at(image_perm[self.cur_iter % len(image_perm)], self.num_pixels)
            rays_o = ray_dict['rays_o']
            rays_d = ray_dict['rays_d'] 
            rays_seg = ray_dict['seg'] 
            mask = ray_dict['mask']  
            true_rgb = ray_dict['color']
            near, far = self.train_dataset.near_far_from_sphere(rays_o, rays_d) 

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            render_out = self.model(rays_o, rays_d, near, far, rays_seg,
                                    cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                    trainstage='Mat')

            loss_output = self.loss(model_output=render_out, object_mask=mask[:,0].bool(),
                                    rgb_gt=true_rgb, mat_model=self.model.envmap_material_network)
            # geo_loss_output = self.geo_loss(model_output=render_out, ground_truth=ray_dict)

            loss = loss_output['sg_total_loss']

            self.neus_mat_optimizer.zero_grad()
            loss.backward()
            self.neus_mat_optimizer.step()

            if self.cur_iter % 10 == 0:
                dr, dt = self.model.obj_poses.get_diff()
                print('[{}]: sg_loss = {:.5f}, psnr = {:.4f}, metal = {:.6f}, kl = {:.6f}, latent={:.6f}, s_val = {:.6f}, lr = {:.6f}, p = {:.3f}, dr = {:.6f}, dt = {:.6f}'.format(
                    self.cur_iter,
                    loss_output['sg_total_loss'].item(),
                    mse2psnr(loss_output['sg_rgb_loss'].item()),
                    loss_output['metal_loss'].item(), 
                    loss_output['kl_loss'].item(),
                    loss_output['latent_smooth_loss'].item(),
                    render_out['s_val'][0].item(), 
                    self.neus_mat_optimizer.param_groups[0]['lr'],
                    progress,
                    dr.mean().item(),
                    dt.mean().item()
                    ))

            if self.cur_iter % len(image_perm) == 0:
                image_perm = torch.randperm(len(self.train_dataset))

            self.cur_iter += 1 

    def evaluate(self):

        print("evaluation...")
        self.model.eval()

        # rgb
        union_mask_psnr0 = []
        obj_mask_psnr0 = []
        union_mask_ssim0 = []
        obj_mask_ssim0 = []
        union_mask_lpips0 = []
        obj_mask_lpips0 = []

        # albedo
        union_mask_psnr1 = []
        union_mask_psnr1_align = []
        obj_mask_psnr1 = []
        obj_mask_psnr1_align = []
        union_mask_ssim1 = []
        union_mask_ssim1_align = []
        obj_mask_ssim1 = []
        obj_mask_ssim1_align = []
        union_mask_lpips1 = []
        union_mask_lpips1_align = []
        obj_mask_lpips1 = []
        obj_mask_lpips1_align = []

        # depth
        # union_mask_mse2 = []
        # obj_mask_mse2 = []

        # normal
        union_mask_abs3 = []
        obj_mask_abs3 = []

        # roughness
        union_mask_mse4 = []
        obj_mask_mse4 = [] 

        print('evaluation number =', len(self.eval_dataset)) 

        eval_batch_size = 2048

        for idx in tqdm(range(len(self.eval_dataset))):
        
            ray_dict = self.eval_dataset.gen_rays_at(idx)
            H, W = self.eval_dataset.img_res
            rays_o = ray_dict['rays_o'].split(eval_batch_size)
            rays_d = ray_dict['rays_d'].split(eval_batch_size) 
            seg_gt = self.eval_dataset.seg_at(idx).cuda()
            rays_seg = seg_gt.reshape(-1, 1).split(eval_batch_size)  

            res = []

            start_time = time.time()
            for rays_o_batch, rays_d_batch, rays_seg_batch in tzip(rays_o, rays_d, rays_seg): 

                near, far = self.eval_dataset.near_far_from_sphere(rays_o_batch, rays_d_batch) 
                out = self.model(rays_o_batch.cuda(),
                                rays_d_batch.cuda(),
                                near.cuda(),
                                far.cuda(), 
                                rays_seg_batch.cuda(),
                                cos_anneal_ratio=self.get_cos_anneal_ratio(), 
                                trainstage='Mat',
                                is_eval=True)
             
                res.append({
                    'points': out['points'].detach(),
                    'normals': out['normals'].detach(),
                    'network_object_mask': out['network_object_mask'].detach(), 
                    'roughness':out['roughness'].detach(),
                    'diffuse_albedo': out['diffuse_albedo'].detach(),
                    'sg_specular_rgb': out['sg_specular_rgb'].detach(),
                    'sg_rgb': out['sg_rgb'].detach(), 
                    'pred_rgb': out['sg_rgb'].detach(), 
                    'vis_shadow': out['vis_shadow'].detach(),
                    'gt_vis_shadow': out['vis_shadow'].detach(), # FIXME
                })

                del out

            total_time = time.time() - start_time
            write_path = os.path.join(os.path.dirname(self.eval_dir_value), 'run_time.txt')
            with open(write_path, "a+") as f:
                f.write('iters [{0}]:time = {1} \n'.format(idx, total_time))

            
            total_pixels = self.img_res[0] * self.img_res[1]
            model_outputs = utils.merge_output(res, total_pixels, batch_size=1) 

            object_mask = self.eval_dataset.mask_at(idx)
            ground_truth = {
                'rgb':self.eval_dataset.image_at(idx)[None],
                # 'depth':self.eval_dataset.depth_at(idx)[None],
                'normal':self.eval_dataset.normal_at(idx)[None],
                'roughness':self.eval_dataset.rough_at(idx)[None],
                'albedo':self.eval_dataset.albedo_at(idx)[None],
                'pose':self.eval_dataset.pose_at(idx)
            }

            assert ground_truth['rgb'].shape[0] == 1

            i = idx 

            pred_rgb = model_outputs['pred_rgb'].cuda().reshape(-1,3)
            object_mask = object_mask.cuda().flatten()
            network_mask = model_outputs['network_object_mask'].cuda().flatten()
            rgb_gt = ground_truth['rgb'].cuda().reshape(-1,3)
            points = model_outputs['points'].cuda().reshape(-1,3) 
            # depth_gt = ground_truth['depth'].cuda().flatten()
            normals = model_outputs['normals'].cuda().reshape(-1,3)
            normals_gt = ground_truth['normal'].cuda().reshape(-1,3)
            roughness = model_outputs['roughness'].cuda().reshape(-1,3)
            roughness_gt = ground_truth['roughness'].cuda().reshape(-1,3)
            pose = ground_truth['pose'][None].cuda()
            diffuse_albedo = model_outputs['diffuse_albedo'].cuda().reshape(-1,3)
            albedo_gt = ground_truth['albedo'].cuda().reshape(-1,3)
            vis_shadow = model_outputs['vis_shadow'].cuda().reshape(-1,3)
            specular_rgb = model_outputs['sg_specular_rgb'].cuda().reshape(-1,3)
            sg_rgb = model_outputs['sg_rgb'].cuda().reshape(-1,3)
       
            # rgb is 0 
            eval_dict = evaluate.evaluate_all(
                pred_rgb=pred_rgb,
                object_mask=object_mask,
                network_mask=network_mask,
                rgb_gt=rgb_gt,
                points=points,
                # depth_gt=depth_gt,
                normals=normals,
                normals_gt=normals_gt,
                roughness=roughness,
                roughness_gt=roughness_gt,
                pose=pose, 
                path=self.eval_dir_value,
                iters=i,
                img_res=self.img_res,
                use_union_mask=True,
                parent_dir=False,
                prefix='rgb')
            union_mask_psnr0.append(eval_dict['psnr'])
            union_mask_ssim0.append(eval_dict['ssim'])
            union_mask_lpips0.append(eval_dict['lpips'])
            # union_mask_mse2.append(eval_dict['depth_mse'])
            union_mask_abs3.append(eval_dict['normal_abs'])
            union_mask_mse4.append(eval_dict['roughness_mse']) 

            eval_dict = evaluate.evaluate_all(
                pred_rgb=pred_rgb,
                object_mask=object_mask,
                network_mask=network_mask,
                rgb_gt=rgb_gt,
                points=points,
                # depth_gt=depth_gt,
                normals=normals,
                normals_gt=normals_gt,
                roughness=roughness,
                roughness_gt=roughness_gt,
                pose=pose, 
                path=self.eval_dir_value,
                iters=i,
                img_res=self.img_res,
                use_union_mask=False,
                parent_dir=False,
                prefix='rgb')
            obj_mask_psnr0.append(eval_dict['psnr'])
            obj_mask_ssim0.append(eval_dict['ssim'])
            obj_mask_lpips0.append(eval_dict['lpips'])
            # obj_mask_mse2.append(eval_dict['depth_mse'])
            obj_mask_abs3.append(eval_dict['normal_abs'])
            obj_mask_mse4.append(eval_dict['roughness_mse']) 

            # albedo is 1
            eval_dict = evaluate.evaluate_all(
                pred_rgb=diffuse_albedo,
                object_mask=object_mask,
                network_mask=network_mask,
                rgb_gt=albedo_gt, 
                path=self.eval_dir_value,
                iters=i,
                img_res=self.img_res,
                use_union_mask=True,
                parent_dir=False, 
                prefix='albedo')
            union_mask_psnr1.append(eval_dict['psnr']) 
            union_mask_ssim1.append(eval_dict['ssim'])
            union_mask_lpips1.append(eval_dict['lpips'])

            eval_dict = evaluate.evaluate_all(
                pred_rgb=diffuse_albedo,
                object_mask=object_mask,
                network_mask=network_mask,
                rgb_gt=albedo_gt, 
                path=self.eval_dir_value,
                iters=i,
                img_res=self.img_res,
                use_union_mask=True,
                parent_dir=False, 
                align_channel=True,
                prefix='albedo_align')
            union_mask_psnr1_align.append(eval_dict['psnr']) 
            union_mask_ssim1_align.append(eval_dict['ssim'])
            union_mask_lpips1_align.append(eval_dict['lpips'])

            align_scale = eval_dict['align_scale']

            eval_dict = evaluate.evaluate_all(
                pred_rgb=diffuse_albedo,
                object_mask=object_mask,
                network_mask=network_mask,
                rgb_gt=albedo_gt, 
                path=self.eval_dir_value,
                iters=i,
                img_res=self.img_res,
                use_union_mask=False,
                parent_dir=False,
                prefix='albedo')
            obj_mask_psnr1.append(eval_dict['psnr']) 
            obj_mask_ssim1.append(eval_dict['ssim'])
            obj_mask_lpips1.append(eval_dict['lpips'])

            eval_dict = evaluate.evaluate_all(
                pred_rgb=diffuse_albedo,
                object_mask=object_mask,
                network_mask=network_mask,
                rgb_gt=albedo_gt, 
                path=self.eval_dir_value,
                iters=i,
                img_res=self.img_res,
                use_union_mask=False,
                parent_dir=False, 
                align_channel=True,
                align_scale=align_scale,
                prefix='albedo_align')
            obj_mask_psnr1_align.append(eval_dict['psnr']) 
            obj_mask_ssim1_align.append(eval_dict['ssim'])
            obj_mask_lpips1_align.append(eval_dict['lpips'])
 
            if i % 5 == 0: 
                plots.plot_neus_mat(
                    normal=normals.reshape([H,W,3]),
                    normal_gt=normals_gt.reshape([H,W,3]),
                    vis_shadow=vis_shadow.reshape([H,W,3]),
                    diffuse_albedo=diffuse_albedo.reshape([H,W,3]),
                    diffuse_albedo_gt=albedo_gt.reshape([H,W,3]),
                    roughness=roughness.reshape([H,W,3]),
                    roughness_gt=roughness_gt.reshape([H,W,3]),
                    metallic=None,
                    metallic_gt=None,
                    specular_rgb=specular_rgb.reshape([H,W,3]),
                    sg_rgb=sg_rgb.reshape([H,W,3]),
                    rgb_gt=rgb_gt.reshape([H,W,3]),
                    pose=pose,
                    # depth_gt=depth_gt.reshape([H,W,1]),
                    points=model_outputs['points'].reshape([1,H*W,3]),
                    obj_mask=object_mask.reshape([H,W,1]),
                    network_obj_mask=network_mask.reshape([H,W,1]),
                    path=self.eval_dir_image,
                    iters=i,
                    align_scale=align_scale,
                )

        union_mask_psnr0 = np.array(union_mask_psnr0)
        obj_mask_psnr0 = np.array(obj_mask_psnr0)
        union_mask_ssim0 = np.array(union_mask_ssim0)
        obj_mask_ssim0 = np.array(obj_mask_ssim0)
        union_mask_lpips0 = np.array(union_mask_lpips0)
        obj_mask_lpips0 = np.array(obj_mask_lpips0)

        union_mask_psnr1_align = np.array(union_mask_psnr1_align)
        obj_mask_psnr1_align = np.array(obj_mask_psnr1_align)
        union_mask_ssim1_align = np.array(union_mask_ssim1_align)
        obj_mask_ssim1_align = np.array(obj_mask_ssim1_align)
        union_mask_lpips1_align = np.array(union_mask_lpips1_align)
        obj_mask_lpips1_align = np.array(obj_mask_lpips1_align)

        union_mask_psnr1 = np.array(union_mask_psnr1)
        obj_mask_psnr1 = np.array(obj_mask_psnr1)
        union_mask_ssim1 = np.array(union_mask_ssim1)
        obj_mask_ssim1 = np.array(obj_mask_ssim1)
        union_mask_lpips1 = np.array(union_mask_lpips1)
        obj_mask_lpips1 = np.array(obj_mask_lpips1)

        # union_mask_mse2 = np.array(union_mask_mse2)
        # obj_mask_mse2 = np.array(obj_mask_mse2)

        union_mask_abs3 = np.array(union_mask_abs3)
        obj_mask_abs3 = np.array(obj_mask_abs3)

        union_mask_mse4 = np.array(union_mask_mse4)
        obj_mask_mse4 = np.array(obj_mask_mse4) 
        
        write_path = os.path.join(os.path.dirname(self.eval_dir_value), 'rgb_mean_value.txt')
        with open(write_path, "a+")as f:
            f.write('# union mask \n')
            f.write('mean psnr = {0} \n'.format(np.mean(union_mask_psnr0)))
            f.write('mean ssim = {0} \n'.format(np.mean(union_mask_ssim0)))
            f.write('mean lpips = {0} \n'.format(np.mean(union_mask_lpips0)))

            f.write('# object mask \n')
            f.write('mean psnr = {0} \n'.format(np.mean(obj_mask_psnr0)))
            f.write('mean ssim = {0} \n'.format(np.mean(obj_mask_ssim0)))
            f.write('mean lpips = {0} \n'.format(np.mean(obj_mask_lpips0)))

        write_path = os.path.join(os.path.dirname(self.eval_dir_value), 'albedo_mean_value.txt')
        with open(write_path, "a+")as f:
            f.write('# union mask \n')
            f.write('mean psnr = {0} \n'.format(np.mean(union_mask_psnr1)))
            f.write('mean ssim = {0} \n'.format(np.mean(union_mask_ssim1)))
            f.write('mean lpips = {0} \n'.format(np.mean(union_mask_lpips1)))

            f.write('# object mask \n')
            f.write('mean psnr = {0} \n'.format(np.mean(obj_mask_psnr1)))
            f.write('mean ssim = {0} \n'.format(np.mean(obj_mask_ssim1)))
            f.write('mean lpips = {0} \n'.format(np.mean(obj_mask_lpips1)))

            f.write('# union mask aligned \n')
            f.write('mean psnr = {0} \n'.format(np.mean(union_mask_psnr1_align)))
            f.write('mean ssim = {0} \n'.format(np.mean(union_mask_ssim1_align)))
            f.write('mean lpips = {0} \n'.format(np.mean(union_mask_lpips1_align)))

            f.write('# object mask aligned \n')
            f.write('mean psnr = {0} \n'.format(np.mean(obj_mask_psnr1_align)))
            f.write('mean ssim = {0} \n'.format(np.mean(obj_mask_ssim1_align)))
            f.write('mean lpips = {0} \n'.format(np.mean(obj_mask_lpips1_align)))

        write_path = os.path.join(os.path.dirname(self.eval_dir_value), 'normal_mean_value.txt')
        with open(write_path, "a+")as f:
            f.write('# union mask \n')
            f.write('mean abs = {0}° \n'.format(np.mean(union_mask_abs3))) 

            f.write('# object mask \n')
            f.write('mean abs = {0}° \n'.format(np.mean(obj_mask_abs3))) 

        write_path = os.path.join(os.path.dirname(self.eval_dir_value), 'roughness_mean_value.txt')
        with open(write_path, "a+")as f:
            f.write('# union mask \n')
            f.write('mean mse = {0} \n'.format(np.mean(union_mask_mse4))) 

            f.write('# object mask \n')
            f.write('mean mse = {0} \n'.format(np.mean(obj_mask_mse4)))  

    def evaluate_envmap(self):
        # log environment map
        lgtSGs = self.model.envmap_material_network.get_light()
        # cur_num = self.model.get_env_sg_num(self.progress, lgtSGs)
        envmap = compute_envmap(lgtSGs=lgtSGs, H=256, W=512, upper_hemi=self.model.envmap_material_network.upper_hemi)
        envmap_eval = tonemap_img(clip_img(envmap)).cpu().numpy().astype(np.float32) 
        envmap_draw = envmap.cpu().numpy()
        imageio.imwrite(os.path.join(os.path.dirname(self.eval_dir_value), 'envmap.exr'), envmap_draw)

        write_path = os.path.join(os.path.dirname(self.eval_dir_value), 'env_map_mse.txt')
        if self.gt_envmap is not None:
            env_mse = np.mean(np.power(self.gt_envmap - envmap_eval, 2))
            with open(write_path, "a+")as f:
                f.write('env mse = {0} \n'.format(env_mse))   
        
    def evaluate_relight(self, envmap_name:str):
        print("evaluation...")
        self.model.eval()

        H = self.img_res[0]
        W = self.img_res[1]

        envmap_path = './envmaps/' + envmap_name
        self.model.envmap_material_network.load_light(envmap_path)

        assert envmap_name in ['b', 'd']

        eval_dir_value = os.path.join(self.expdir, self.timestamp, envmap_name, 'evals_value')
        eval_dir_image = os.path.join(self.expdir, self.timestamp, envmap_name, 'evals_image')
        self.eval_dir_value = eval_dir_value
        self.eval_dir_image = eval_dir_image
        utils.mkdir_ifnotexists(eval_dir_value)
        utils.mkdir_ifnotexists(eval_dir_image) 
 
        # rgb
        union_mask_psnr0 = []
        obj_mask_psnr0 = []
        union_mask_ssim0 = []
        obj_mask_ssim0 = []
        union_mask_lpips0 = []
        obj_mask_lpips0 = []

        print('evaluation number =', len(self.eval_dataset)) 

        eval_batch_size = 2048 

        for idx in tqdm(range(len(self.eval_dataset))):
        
            ray_dict = self.eval_dataset.gen_rays_at(idx)
            H, W = self.eval_dataset.img_res
            rays_o = ray_dict['rays_o'].split(eval_batch_size)
            rays_d = ray_dict['rays_d'].split(eval_batch_size)  
            seg_gt = self.eval_dataset.seg_at(idx).cuda()
            rays_seg = seg_gt.reshape(-1, 1).split(eval_batch_size)
            res = []

            start_time = time.time()
            for rays_o_batch, rays_d_batch, rays_seg_batch in tzip(rays_o, rays_d, rays_seg): 

                near, far = self.eval_dataset.near_far_from_sphere(rays_o_batch, rays_d_batch )

                out = self.model(rays_o_batch.cuda(),
                                rays_d_batch.cuda(),
                                near.cuda(),
                                far.cuda(), 
                                rays_seg_batch.cuda(),
                                perturb_overwrite=0,
                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                trainstage='Mat',
                                is_eval=True)
             
                res.append({ 
                    'sg_rgb': out['sg_rgb'].detach(),  
                    'network_object_mask': out['network_object_mask'].detach(), 
                })

                del out

            total_time = time.time() - start_time
            write_path = os.path.join(os.path.dirname(self.eval_dir_value), 'run_time.txt')
            with open(write_path, "a+") as f:
                f.write('iters [{0}]:time = {1} \n'.format(idx, total_time))
            
            total_pixels = self.img_res[0] * self.img_res[1]
            model_outputs = utils.merge_output(res, total_pixels, batch_size=1) 

            object_mask = self.eval_dataset.mask_at(idx)
            ground_truth = {
                'rgb':self.eval_dataset.image_at(idx, envmap_name=envmap_name)[None] 
            }

            assert ground_truth['rgb'].shape[0] == 1

            i = idx 

            pred_rgb = model_outputs['sg_rgb'].cuda().reshape(-1,3)
            object_mask = object_mask.cuda().flatten()
            network_mask = model_outputs['network_object_mask'].cuda().flatten()
            rgb_gt = ground_truth['rgb'].cuda().reshape(-1,3) 
       
            # rgb is 0 
            eval_dict = evaluate.evaluate_all(
                pred_rgb=pred_rgb,
                object_mask=object_mask,
                network_mask=network_mask,
                rgb_gt=rgb_gt,
                path=self.eval_dir_value,
                iters=i,
                img_res=self.img_res,
                use_union_mask=True,
                parent_dir=False,
                prefix='rgb')
            union_mask_psnr0.append(eval_dict['psnr'])
            union_mask_ssim0.append(eval_dict['ssim'])
            union_mask_lpips0.append(eval_dict['lpips'])

            eval_dict = evaluate.evaluate_all(
                pred_rgb=pred_rgb,
                object_mask=object_mask,
                network_mask=network_mask,
                rgb_gt=rgb_gt,
                path=self.eval_dir_value,
                iters=i,
                img_res=self.img_res,
                use_union_mask=False,
                parent_dir=False,
                prefix='rgb')
            obj_mask_psnr0.append(eval_dict['psnr'])
            obj_mask_ssim0.append(eval_dict['ssim'])
            obj_mask_lpips0.append(eval_dict['lpips']) 
 
            if i % 5 == 0: 
                plots.plot_rgb(
                    {'sg_rgb':pred_rgb.reshape(1, H*W, 3),
                    'object_mask':object_mask,
                    'network_object_mask':network_mask},
                    rgb_gt.reshape(1, H*W, 3).cuda(),
                    eval_dir_image,
                    i,
                    self.img_res,
                    name=self.name) 

        union_mask_psnr0 = np.array(union_mask_psnr0)
        obj_mask_psnr0 = np.array(obj_mask_psnr0)
        union_mask_ssim0 = np.array(union_mask_ssim0)
        obj_mask_ssim0 = np.array(obj_mask_ssim0)
        union_mask_lpips0 = np.array(union_mask_lpips0)
        obj_mask_lpips0 = np.array(obj_mask_lpips0)
        
        write_path = os.path.join(os.path.dirname(self.eval_dir_value), 'rgb_mean_value.txt')
        with open(write_path, "a+")as f:
            f.write('# union mask \n')
            f.write('mean psnr = {0} \n'.format(np.mean(union_mask_psnr0)))
            f.write('mean ssim = {0} \n'.format(np.mean(union_mask_ssim0)))
            f.write('mean lpips = {0} \n'.format(np.mean(union_mask_lpips0)))

            f.write('# object mask \n')
            f.write('mean psnr = {0} \n'.format(np.mean(obj_mask_psnr0)))
            f.write('mean ssim = {0} \n'.format(np.mean(obj_mask_ssim0)))
            f.write('mean lpips = {0} \n'.format(np.mean(obj_mask_lpips0)))