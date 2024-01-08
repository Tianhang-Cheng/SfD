import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from termcolor import colored
from functools import partial

import warnings

from model.geometry_mlp import ImplicitNetwork 
from model.material_sg import EnvmapMaterialNetwork
from model.visibility import SirenVisNetwork, VisNetwork
from model.embedder import get_embedder
from model.material_sg import render_with_all_sg
from model.pose import ObjectPose
from model.proxycolor import FourierColorNetwork

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]).cuda(), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).cuda()
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).cuda()

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1).cuda(), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds).cuda(), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom).cuda(), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class Regression(nn.Module):
    def __init__(self, h_dim, n_layers, multires=6):
        super(Regression, self).__init__()  

        input_dim = 3
        self.embed_fn = None
        if multires > 0:
            self.embed_fn, input_dim = get_embedder(multires)
        self.actv_fn = nn.ReLU()
 
        linears = []
        dim = input_dim
        for _ in range(n_layers-1):
            linears.append(nn.Linear(dim, h_dim))
            linears.append(self.actv_fn)
            dim = h_dim
        linears.append(nn.Linear(dim, 1))
        self.linears = nn.Sequential(*linears)

        self.seg = None 

    def forward(self, x_world, obj_poses, direct=True, use_grad=True): 
        """
        x_world: [b,3]   

        gt_class: [b, 1]
        pred_class: [b, 1]
        """

        assert x_world.shape[-1] == 3, 'shape = {0}'.format(x_world.shape[-1])
        assert len(x_world.shape) == 2, 'length = {0}'.format(len(x_world.shape)) 
        

        with torch.set_grad_enabled(use_grad):

            inv_poses = obj_poses.get_inv_pose() # [n, 4, 4] world to object
            poses = obj_poses.get_pose() # [n, 4, 4] object to world 
            vector = obj_poses.get_variant_vector() # [n, dim] 

            b = x_world.shape[0]
            n = inv_poses.shape[0] 

            x_local_all = torch.einsum('nxy,by->nbx', inv_poses[:, 0:3,0:3], x_world) + inv_poses[:, None, 0:3, 3]   # world to local

            if self.seg is None and direct:
                warnings.warn('### No segmentation but use direct mode, Use this mode only for visibility training. ###')
            if direct:
                warnings.warn('### Use direct mode for fast inferennce ###') 

            pred_prob = None 

            # return classifcation result by query segmentation image
            if self.seg is not None and direct:
                pred_class = self.seg.clone()
                pred_class = pred_class - 1
                pred_class[pred_class == -1] = np.random.randint(0, n)
                if pred_class.shape[0] > b:
                    import pdb
                    pdb.set_trace()
                    raise ValueError
                if pred_class.shape[0] != b:
                    n_sample = b // pred_class.shape[0]
                    pred_class = torch.repeat_interleave(pred_class, dim=1, repeats=n_sample)
                    pred_class = pred_class.reshape([-1, 1])
            
            # return classifcation result by query neural network
            else:
                # pred_prob = self.linears(self.embed_fn(x_world)).reshape([b, n]) 
                split_num = n // 20 + 1
                pred_prob = []
                for i,idx in enumerate(torch.split(torch.arange(n).cuda(), n//split_num, dim=0)):
                    dn = len(idx) 
                    x_local_all_ = torch.index_select(x_local_all, dim=0, index=idx).reshape([dn*b, 3]) 
                    pred_prob_ = self.linears(self.embed_fn(x_local_all_)).reshape([dn, b])
                    # pred_prob_ = self.linears(x_local_all_).reshape([dn, b])
                    pred_prob.append(pred_prob_)
                pred_prob = torch.cat(pred_prob, dim=0) # [n, b]
                pred_class = torch.argmin(pred_prob, dim=0, keepdim=False).unsqueeze(1)  # b,1
                pred_prob = pred_prob.permute([1,0])   # b, n 
            min_encodings = torch.zeros(pred_class.shape[0], inv_poses.shape[0]).cuda()  # b,n
            min_encodings.scatter_(1, pred_class, 1)

            inv_pose_q = torch.einsum('bn,nxy->bxy', min_encodings, inv_poses)  # world to obj
            vec_q = torch.einsum('bn,nk->bk', min_encodings, vector) if vector is not None else None  # world to obj
            # pose_q = torch.einsum('bn,nxy->bxy', min_encodings, poses) # obj to world   
            
        x_local = torch.einsum('bxy,by->bx',inv_pose_q[:, 0:3, 0:3], x_world) + inv_pose_q[:, 0:3, 3] # [b,3]
 
        res = {'x_local': x_local,
                'x_local_all': x_local_all,
                'x_world': x_world,
                'min_encodings': min_encodings, 
                'inv_pose_q': inv_pose_q,
                'pred_class':pred_class,
                'pred_prob':pred_prob,
                'vec_q': vec_q,
                }
        
        return res  

    def get_pred_prob(self, x_world, obj_poses):
        """  
        return:
            proxy_sdf [batch, n_class]
        """   
        res = self.forward(x_world, obj_poses=obj_poses, direct=False)
        pred_prob = res['pred_prob'] 
        x_local_all = res['x_local_all']
        return pred_prob, x_local_all

    def supervise(self, x_world, obj_poses, sdf_network):
        """ 
        network_sdf [b, c]
        proxy_sdf [b, c]
        """  
        proxy_sdf, x_local_all = self.get_pred_prob(x_world, obj_poses=obj_poses)

        # network sdf supervise

        variant_vectors = obj_poses.get_variant_vector()
        if variant_vectors is not None:
            vector = variant_vectors.mean(0, keepdims=True)
        
        n,b,_ = x_local_all.shape
        split_num = n // 5 + 1

        network_sdf = []
        for _, idx in enumerate(torch.split(torch.arange(n).cuda(), n//split_num, dim=0)):
            dn = len(idx)
            x_local_all_temp = torch.index_select(x_local_all, dim=0, index=idx).reshape([dn*b, 3]) 
            if variant_vectors is None:  
                sdf_output_temp = sdf_network(x_local_all_temp)[..., 0].reshape([dn, b])
            else:
                sdf_output_temp = sdf_network(x_local_all_temp, input_variant_vector=vector.expand([dn*b, -1]))[..., 0].reshape([dn, b])
            network_sdf.append(sdf_output_temp)
        network_sdf = torch.cat(network_sdf, dim=0).permute([1,0]) # [batch, n_class]  

        return proxy_sdf, network_sdf   
    
class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]).cuda() * torch.exp(self.variance * 10.0)

class DupNeuSRenderer(nn.Module):
    def __init__(self,
                 same_obj_num,
                 visible_num,
                 init_method,
                 data_split_dir,
                 scene_radius,
                 scene_center,
                 conf,
                 real_world,
                 train_pose=False,
                 **kwargs
                 ):
        super().__init__() 
        self.feature_vector_size = conf.get('feature_vector_size')
        self.variant_vector_size = conf.get('variant_vector_size', 0)
        self.same_obj_num = same_obj_num
        self.visible_num = visible_num
        self.n_samples = conf.get('n_samples')
        self.n_importance = conf.get('n_importance')
        self.n_outside = conf.get('n_outside')
        self.up_sample_steps = conf.get('up_sample_steps')
        self.perturb = conf.get('perturb') 
        self.train_pose = train_pose

        self.weight_threshold = conf.get('weight_threshold', 0.5) 
        self.scene_radius = scene_radius
        self.scene_center = scene_center

        # ablation study
        self.use_numerical_gradient = conf.get('use_numerical_gradient', False)
        if self.use_numerical_gradient:
            print(colored('Using numerical gradient for sdf gradient computation!', 'red', attrs=['bold']))
        self.use_colmap_constraint = kwargs.get('use_colmap_constraint', None)
        if self.use_colmap_constraint is None:
            self.use_colmap_constraint = conf.get('use_colmap_constraint', False)
        if self.use_colmap_constraint:
            print(colored('Using colmap constraint for geometry surface!', 'red', attrs=['bold']))
            self.colmap_point = kwargs.get('colmap_point', None)
            assert self.colmap_point is not None, 'colmap_point should be provided when use_colmap_constraint is True!'
        self.use_hessian = conf.get('use_hessian', False)
        if self.use_hessian:
            print(colored('Using hessian for geometry constrain!', 'red', attrs=['bold']))
        self.use_triplane = conf.get('use_triplane', False)
        if self.use_triplane:
            from model.geometry_triplane import TriPlaneGenerator
            print(colored('Using triplane as geometry representation!', 'red', attrs=['bold']))
        self.use_hash = conf.get('use_hash', False)
        if self.use_hash:
            from model.geometry_hash import HashNeuralSDF
            print(colored('Using hash MLP as geometry representation!', 'red', attrs=['bold']))

        assert self.same_obj_num >= 1
        assert self.scene_radius > 0

        if self.use_triplane:
            self.sdf_network = TriPlaneGenerator(**conf.get('triplane_network'))
        elif self.use_hash:
            self.sdf_network = HashNeuralSDF(conf.get('hash_network'))
        else:
            self.sdf_network = ImplicitNetwork(
                feature_vector_size=self.feature_vector_size,
                variant_vector_size=self.variant_vector_size,
                use_numerical_gradient=self.use_numerical_gradient,
                use_hessian=self.use_hessian,
                **conf.get('implicit_network')
            )
        self.deviation_network = SingleVarianceNetwork(**conf.get('variance_network'))
        self.rendering_network = FourierColorNetwork(self.feature_vector_size, **conf.get('rendering_network'))
 
        self.envmap_material_network = EnvmapMaterialNetwork(**conf.get('envmap_material_network'))
 
        self.use_siren_vis = conf.get('use_siren_vis', False)
        if self.use_siren_vis:
            self.visibility_network = SirenVisNetwork(**conf.get('visibility_network'))
        else:
            self.visibility_network = VisNetwork(**conf.get('visibility_network'))

        self.point_classify_network = Regression(**conf.get('transformation_network_regress'))

        self.obj_poses = ObjectPose(same_obj_num=same_obj_num,
                                    visible_num=visible_num,
                                    train_pose=self.train_pose,
                                    data_split_dir=data_split_dir,
                                    init_method=init_method, 
                                    real_world=real_world,
                                    variant_vector_size=self.variant_vector_size)

        self.classify = partial(self.point_classify_network, obj_poses=self.obj_poses)

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts - self.scene_center, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < self.scene_radius) | (radius[:, 1:] < self.scene_radius)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]).cuda(), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere
        cos_val = cos_val.clip(-1e3, 0.0)

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]).cuda(), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)  

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3), self.classify).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    rendering_network,
                    cos_anneal_ratio=0.0,
                    is_training_geometry=False,
                    is_eval=False,
                    seg=None):
        is_training = not is_eval
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).cuda().expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts_ = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts_.shape)
        pts = pts_.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3) 

        
        if is_eval:
            prob_q = self.classify(pts, direct=False, use_grad=False)['pred_prob'] # jsut for draw, no need to re-center because cast to object space
        else:
            prob_q = torch.zeros([pts.shape[0], self.visible_num]).cuda() # placeholder, all zero
        
        colmap_sdf_error = torch.tensor(0.0).float().cuda()
        if self.use_colmap_constraint:
            inv_pose = self.obj_poses.get_inv_pose()[0]
            colmap_point_local = (inv_pose[0:3, 0:3] @ self.colmap_point.T + inv_pose[0:3, 3:4]).T
            colmap_sdf = self.sdf_network.sdf(colmap_point_local)
            colmap_sdf_error = torch.abs(colmap_sdf).mean()

        trans_res = self.classify(pts, direct=True) 
        pts_local = trans_res['x_local'] 
        inv_pose_q = trans_res['inv_pose_q']
        del trans_res

        dirs_local = torch.einsum('bxy,by->bx',inv_pose_q[:, 0:3, 0:3], dirs) # [N,3,3] @ [N,3,1]
        dirs_local = dirs_local / (torch.norm(dirs_local, dim=1, keepdim=True)+1e-6)  

        with torch.set_grad_enabled(is_training_geometry):
            sdf_nn_output = sdf_network(pts, transform_func=self.classify) # no need to re-center because cast to object space
            sdf = sdf_nn_output[:, :1]
            feature_vector = sdf_nn_output[:, 1:]
        
        # gradient function returns world gradient, not local; no need to re-center because cast to object space
        res = sdf_network.gradient(pts.clone(), transform_func=self.classify, is_training=is_training, is_training_geometry=is_training_geometry, sdf=sdf) 
        gradients_world = res['gradients_world']
        hessian = res['hessian']
        normals_world = gradients_world / (torch.norm(gradients_world, dim=1, keepdim=True)+1e-6)
        normals_local = torch.einsum('bxy,by->bx',inv_pose_q[:, 0:3, 0:3], normals_world) # [N,3,3] @ [N,3] -> [N,3]
        normals_local = normals_local / (torch.norm(normals_local, dim=1, keepdim=True)+1e-6)

        with torch.set_grad_enabled(is_training_geometry):

            inv_s = deviation_network(torch.zeros([1, 3]).cuda())[:, :1].clip(1e-6, 1e6)   # Single parameter
            inv_s = inv_s.expand(batch_size * n_samples, 1) 

            true_cos = (dirs * normals_world).sum(-1, keepdim=True)

            # pdb.set_trace()

            # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
            # the cos value "not dead" at the beginning training iterations, for better convergence.
            iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
            # Estimate signed distances at section points
            estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
            estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

            # pdb.set_trace()
            pts_norm = torch.linalg.norm(pts - self.scene_center, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples) #  need to re-center to calculate real radius
            inside_sphere = (pts_norm < self.scene_radius).float().detach()
            relax_inside_sphere = (pts_norm < self.scene_radius * 1.2).float().detach() 

            weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]).cuda(), 1. - alpha + 1e-7], -1), -1)[:, :-1]
            # Surface points
            surface_points = ((pts_) * weights[..., None]).sum(dim=1)  # n_rays, 3
            surface_points_local = (pts_local.reshape(batch_size, n_samples, 3) * weights[..., None]).sum(dim=1)  # n_rays, 3

            weights_sum = weights.sum(dim=-1, keepdim=True)
            network_object_mask = (weights_sum > self.weight_threshold)[:,0]
              
            surface_normals_local = normals_local.reshape(batch_size, n_samples, 3)
            surface_normals_local = (surface_normals_local * weights[:, :, None]).sum(dim=1) 
            surface_normals_local = surface_normals_local / (torch.norm(surface_normals_local, dim=1, keepdim=True)+1e-6)
            surface_normals_world = normals_world.reshape(batch_size, n_samples, 3)
            surface_normals_world = surface_normals_world * inside_sphere[..., None]
            surface_normals_world = (surface_normals_world * weights[:, :, None]).sum(dim=1) 
            surface_normals_world = surface_normals_world / (torch.norm(surface_normals_world, dim=1, keepdim=True)+1e-6)

            hessian_error = torch.tensor(0.0).float().cuda()
            if hessian is not None:
                surface_hessian = hessian.reshape(batch_size, n_samples, 3)
                surface_hessian = surface_hessian * inside_sphere[..., None]
                surface_hessian = (surface_hessian * weights[:, :, None]).sum(dim=1)
                hessian_error = surface_hessian.sum(dim=-1).abs()
                hessian_error = hessian_error.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                hessian_error = hessian_error.mean()

            prob_q = prob_q.reshape(batch_size, n_samples, self.visible_num)
            proxy_class_prob = (prob_q * weights[:, :, None]).sum(dim=1)
            proxy_pred_class = torch.argmin(proxy_class_prob, dim=1, keepdim=True)  # b,1  
            # surf_pose_q = torch.index_select(self.obj_poses.get_pose(), dim=0, index=proxy_pred_class[:,0])
            # surf_inv_pose_q = torch.index_select(self.obj_poses.get_pose(), dim=0, index=proxy_pred_class[:,0]

            gradient_error = (torch.linalg.norm(gradients_world.reshape(batch_size, n_samples, 3), ord=2, dim=-1) - 1.0) ** 2
            gradient_error = gradient_error.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

            if is_training_geometry:
                render_dict = rendering_network(pts_local, normals_local, dirs_local, feature_vector, pts - self.scene_center, normals_world, dirs)
                sampled_color = render_dict['color'].reshape(batch_size, n_samples, 3)   
                color = (sampled_color * weights[:, :, None]).sum(dim=1) 
                sampled_albedo = render_dict['albedo'].reshape(batch_size, n_samples, 3)   
                albedo = (sampled_albedo * weights[:, :, None]).sum(dim=1)
            else:
                color = None
                albedo = None

            select_candidate = []
            if seg is not None:
                select_candidate = torch.where(seg > 0)[0]
            if len(select_candidate) == 0 or is_eval:
                select_gt_seg = None
                select_proxy_sdf = None
                select_network_sdf = None
            else:
                selected_indices = select_candidate[torch.randperm(len(select_candidate))[:64]]
                select_proxy_sdf, select_network_sdf = self.point_classify_network.supervise(pts_[selected_indices].reshape(-1, 3), self.obj_poses, self.sdf_network)  # for train, input [128, 64, 3]
                select_proxy_sdf = torch.sum(weights[selected_indices][..., None].detach() * select_proxy_sdf.reshape(len(selected_indices), -1, self.visible_num), dim=1)
                select_network_sdf = torch.sum(weights[selected_indices][..., None] * select_network_sdf.reshape(len(selected_indices), -1, self.visible_num), dim=1)
                select_gt_seg = seg[selected_indices] - 1 # minus background
 
            return {
                'pts': pts,
                'sdf': sdf,
                's_val': (1.0 / inv_s).reshape(batch_size, n_samples).mean(dim=-1, keepdim=True),
                'color': color,
                'albedo': albedo,
                'alpha': alpha,
                'weights': weights,
                'weights_sum': weights_sum,
                'network_object_mask': network_object_mask,
                'gradients': normals_world.reshape(batch_size, n_samples, 3),
                'gradient_error': gradient_error,
                'hessian_error': hessian_error,
                'colmap_sdf_error': colmap_sdf_error,
                'inside_sphere': inside_sphere,

                'z_vals':mid_z_vals,
                'dists':dists,

                'surface_points': surface_points, 
                'surface_points_local': surface_points_local,
                'surface_normal': surface_normals_world,
                'surface_normal_local': surface_normals_local,
                # 'surface_feature': surf_feature, 
                # 'surface_sdf': None, 
                # 'surface_proxy_class_prob':proxy_class_prob,
                'surface_proxy_pred_class':proxy_pred_class,

                'select_proxy_sdf': select_proxy_sdf, 
                'select_network_sdf': select_network_sdf,
                'select_gt_seg': select_gt_seg,
            }
    
    def trace_visibility(self, rays_o, rays_d, near, far, seg, perturb_overwrite=-1, cos_anneal_ratio=0.0, vis_nsamp=4):

        ret_fine = self.forward(rays_o, rays_d, near, far, seg,
                                perturb_overwrite=perturb_overwrite,
                                cos_anneal_ratio=cos_anneal_ratio,
                                trainstage='Vis',
                                is_eval=True)
        
        weights_sum = ret_fine['weights_sum']
        surface_points_world = ret_fine['surface_points_world']
        surface_normal_world = ret_fine['surface_normal_world']
        network_object_mask = ret_fine['network_object_mask']  

        gt_vis = torch.zeros(weights_sum.shape[0], vis_nsamp, 1).bool().cuda()
        pred_vis = torch.zeros(weights_sum.shape[0], vis_nsamp, 2).cuda()

        sec_cam_loc = surface_points_world[network_object_mask].clone() # world
        sec_surf_normal = surface_normal_world[network_object_mask].clone().detach()[:, None]

        if sec_cam_loc.shape[0] > 0:
            r_theta = torch.rand(sec_cam_loc.shape[0], vis_nsamp).cuda() * 2 * np.pi
            rand_z = torch.rand(sec_cam_loc.shape[0], vis_nsamp).cuda() * 0.95
            r_phi = torch.asin(rand_z)  
            sample_dirs = self.sample_dirs(sec_surf_normal, r_theta, r_phi)
            input_p = sec_cam_loc.unsqueeze(1).expand(-1, vis_nsamp, 3).reshape(-1, 3)

            scene_diameter = self.scene_radius * 2

            sec_net_object_mask = self.forward(rays_o=input_p, rays_d=sample_dirs.reshape(-1, 3),
                                                near=torch.ones([input_p.shape[0], 1]).cuda() * scene_diameter * 0.02,
                                                far=torch.ones([input_p.shape[0], 1]).cuda() * scene_diameter,
                                                seg=None,
                                                perturb_overwrite=0, cos_anneal_ratio=cos_anneal_ratio,
                                                trainstage='Vis')['weights_sum']
            sec_net_object_mask = (sec_net_object_mask > 0.95)[:,0]

            gt_vis[network_object_mask] = sec_net_object_mask.reshape(sec_cam_loc.shape[0], vis_nsamp, 1)
            
            query_vis = self.visibility_network(input_p, sample_dirs.reshape(-1, 3))
            pred_vis[network_object_mask] = query_vis.reshape(-1, vis_nsamp, 2)
            
        result_dict = { 
            'network_object_mask': network_object_mask,
            'gt_vis': gt_vis,
            'pred_vis': pred_vis
            }
        return result_dict
    
    def sample_dirs(self, normals, r_theta, r_phi):
        TINY_NUMBER = 1e-6
        z_axis = torch.zeros_like(normals).cuda()
        z_axis[:, :, 0] = 1

        def norm_axis(x):
            return x / (torch.norm(x, dim=-1, keepdim=True) + TINY_NUMBER)

        normals = norm_axis(normals)
        U = norm_axis(torch.cross(z_axis, normals))
        V = norm_axis(torch.cross(normals, U))

        r_theta = r_theta.unsqueeze(-1).expand(-1, -1, 3)
        r_phi = r_phi.unsqueeze(-1).expand(-1, -1, 3)
        sample_raydirs = U * torch.cos(r_theta) * torch.sin(r_phi) \
                        + V * torch.sin(r_theta) * torch.sin(r_phi) \
                        + normals * torch.cos(r_phi) # [num_cam, num_samples, 3]
        return sample_raydirs
     
    def get_pred_visibility(self, masked_points, masked_normals, nsamp=8): 
          
        sample_dirs = torch.zeros(masked_points.shape[0], nsamp, 3).cuda()
        pred_vis = torch.zeros(masked_points.shape[0], nsamp, 2).cuda() 

        if masked_points.shape[0] > 0: 
            r_theta = torch.rand(masked_points.shape[0], nsamp).cuda() * 2 * np.pi
            rand_z = torch.rand(masked_points.shape[0], nsamp).cuda()
            r_phi = torch.asin(rand_z)
            sample_dirs = self.sample_dirs(masked_normals.detach()[:, None], r_theta, r_phi) 
  
            input_p = masked_points.unsqueeze(1).expand(-1, nsamp, 3)
            with torch.no_grad():
                pred_vis = self.visibility_network(input_p.reshape(-1, 3), sample_dirs.reshape(-1, 3)).reshape(-1, nsamp, 2) 
        
        pred_vis = F.softmax(pred_vis.detach(), dim=-1)[..., 1]
        # pred_vis = torch.max(pred_vis, dim=-1)[1].float()
        
        pred_vis = torch.mean(pred_vis.float(), dim=1)
                       
        return pred_vis

    def forward(self, rays_o, rays_d, near, far, seg, perturb_overwrite=-1, cos_anneal_ratio=0.0, trainstage='Geo', is_eval=False):

        assert trainstage in ['Geo','Vis','Mat']
        if trainstage != 'Vis':
            assert seg is not None
        self.point_classify_network.seg = seg   

        batch_size = len(rays_o)
        sample_dist = (far[0] - near[0]) / self.n_samples   # Assuming the region of interest is a sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples).cuda()
        z_vals = near + (far - near) * z_vals[None, :]  
        
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5).cuda()
            z_vals = z_vals + t_rand * (2 * self.scene_radius) / self.n_samples 

        # Up sample
        if self.n_importance > 0:
            with torch.inference_mode():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3), self.classify).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2 ** i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                rays_d,
                                                z_vals,
                                                new_z_vals,
                                                sdf,
                                                last=(i + 1 == self.up_sample_steps)) 

        # Render core 
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.rendering_network,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    is_training_geometry=(trainstage=='Geo'), # eval gradient only when training geometry
                                    is_eval=is_eval,
                                    seg=seg)
         
        weights_sum = ret_fine['weights_sum']
        surface_points_local = ret_fine['surface_points_local']
        surface_points_world = ret_fine['surface_points'] 
        surface_normal_world = ret_fine['surface_normal']

        if trainstage == 'Geo':
             
            result_dict = ret_fine

            poses = self.obj_poses.get_pose() # [b, 4, 4] 
            surface_points_world_all = torch.einsum('nxy, by -> nbx', poses[:, 0:3,0:3], surface_points_local) + poses[:, None, 0:3, 3] 
            surface_normal_world_all = torch.einsum('nxy, by -> nbx', poses[:, 0:3,0:3], surface_normal_world) 

            result_dict.update({
                'surface_points_world_all': surface_points_world_all, 
                'surface_normal_world_all': surface_normal_world_all 
            }) 

            return result_dict
        
        if trainstage == 'Vis':  
            
            result_dict = {
                'weights_sum': weights_sum,
                'network_object_mask': (weights_sum > self.weight_threshold)[:,0],
                'surface_points_world': surface_points_world,
                'surface_normal_world': surface_normal_world,
                }
            
            return result_dict

        if trainstage == 'Mat':

            network_object_mask = (weights_sum > self.weight_threshold)[:,0]
            network_object_mask_draw = network_object_mask
 
            sg_rgb_values = torch.ones_like(surface_points_world).float().cuda() 
            sg_diffuse_rgb_values = torch.ones_like(surface_points_world).float().cuda()
            sg_specular_rgb_values = torch.ones_like(surface_points_world).float().cuda()
            bg_rgb_values = torch.ones_like(surface_points_world).float().cuda()

            normal_values = torch.ones_like(surface_points_world).float().cuda()
            diffuse_albedo_values = torch.ones_like(surface_points_world).float().cuda()
            roughness_values = torch.ones_like(surface_points_world).float().cuda()
            metallic_values = torch.ones_like(surface_points_world).float().cuda()

            vis_shadow = torch.ones_like(surface_points_world).float().cuda()

            random_xi_diffuse_albedo = torch.ones_like(surface_points_world).float().cuda()
            random_xi_roughness = torch.ones_like(surface_points_world).float().cuda()

            diff_surface_points_local = surface_points_local[network_object_mask]
            diff_surface_points_world = surface_points_world[network_object_mask]
            diff_surface_normal_world = surface_normal_world[network_object_mask]

            if diff_surface_points_world.shape[0] > 0:   

                material_network = self.envmap_material_network(diff_surface_points_local)
                sg_roughness = material_network['sg_roughness']
                sg_metallic = material_network['sg_metallic']

                with torch.set_grad_enabled(not is_eval):
                    sg_ret = render_with_all_sg(points=diff_surface_points_world,
                                                normal=diff_surface_normal_world, 
                                                viewdirs=-rays_d[network_object_mask],  # ----> camera , 
                                                lgtSGs=material_network['sg_lgtSGs'],
                                                specular_reflectance=material_network['sg_specular_reflectance'],
                                                roughness=sg_roughness,
                                                metallic=sg_metallic,
                                                diffuse_albedo=material_network['sg_diffuse_albedo'],
                                                indir_lgtSGs=None,
                                                VisModel=self.visibility_network) 

                sg_rgb_values[network_object_mask] = sg_ret['sg_rgb'] 
                sg_diffuse_rgb_values[network_object_mask] = sg_ret['sg_diffuse_rgb']
                sg_specular_rgb_values[network_object_mask] = sg_ret['sg_specular_rgb']

                normal_values[network_object_mask] = diff_surface_normal_world
                normal_values = surface_normal_world
                diffuse_albedo_values[network_object_mask] = material_network['sg_diffuse_albedo']
                roughness_values[network_object_mask] = material_network['sg_roughness'].expand(-1, 3)
                metallic_values[network_object_mask] = material_network['sg_metallic'].expand(-1, 3)
                
                vis_shadow[network_object_mask] = \
                    self.get_pred_visibility(diff_surface_points_world, diff_surface_normal_world).unsqueeze(-1).expand(-1, 3) 

                random_xi_diffuse_albedo[network_object_mask] = material_network['random_xi_diffuse_albedo']
                random_xi_roughness[network_object_mask] = material_network['random_xi_roughness'].expand(-1, 3)
        
            result_dict = {
                'bg_rgb': bg_rgb_values,
                'sg_rgb': sg_rgb_values, 
                'sg_diffuse_rgb': sg_diffuse_rgb_values,
                'sg_specular_rgb': sg_specular_rgb_values,
                'normals': normal_values,
                'diffuse_albedo': diffuse_albedo_values,
                'roughness': roughness_values,
                'metallic': metallic_values,
                'vis_shadow': vis_shadow,
                'points':surface_points_world,
                'points_local':surface_points_local,
                'weights_sum': ret_fine['weights_sum'],
                'gradient_error':ret_fine['gradient_error'],
                'surface_normal':ret_fine['surface_normal'],
                's_val':ret_fine['s_val'],
                'network_object_mask': network_object_mask,
                'network_object_mask_draw': network_object_mask_draw,
                'random_xi_roughness': random_xi_roughness,
                'random_xi_diffuse_albedo': random_xi_diffuse_albedo, 
            }
            return result_dict