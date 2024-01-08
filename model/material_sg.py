
import os
import sys 
import imageio
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math import ceil
from model.embedder import get_embedder

TINY_NUMBER = 1e-6 

def fibonacci_sphere(samples=1):
    '''
    uniformly distribute points on a sphere
    reference: https://github.com/Kai-46/PhySG/blob/master/code/model/material_network.py
    '''
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1 + 1e-5)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])
    points = np.array(points)
    return points


def compute_energy(lgtSGs):
    lgtLambda = torch.abs(lgtSGs[:, 3:4]) 
    lgtMu = torch.abs(lgtSGs[:, 4:]) 
    energy = lgtMu * 2.0 * np.pi / lgtLambda * (1.0 - torch.exp(-2.0 * lgtLambda))
    return energy


class EnvmapMaterialNetwork(nn.Module):
    def __init__(self, multires=0, 
                 brdf_encoder_dims=[512, 512, 512, 512],
                 brdf_decoder_dims=[128, 128],
                 num_lgt_sgs=32,
                 upper_hemi=False,
                 specular_albedo=0.02,
                 latent_dim=32,
                 metallic_range=[0.0, 0.0],
                 albedo_range=[0.117, 0.941],
                 roughness_range=[0.0, 1.0]):
        super().__init__()

        self.default_specular_albedo = specular_albedo 
        
        self.embed_fn = None
        brdf_input_dim = 3
        if multires > 0:
            self.brdf_embed_fn, brdf_input_dim = get_embedder(multires)
        self.brdf_input_dim = brdf_input_dim
 
 
        self.numLgtSGs = num_lgt_sgs
        self.envmap = None
        self.metallic_range = metallic_range
        self.albedo_range = np.power(albedo_range, 2.2)
        self.roughness_range = roughness_range

        self.latent_dim = latent_dim
        self.actv_fn = nn.LeakyReLU(0.2) 
        ############## spatially-varying BRDF ############

        self.specular_reflectance = nn.Parameter(torch.zeros([1,1]).cuda(), requires_grad=False)
        
        print('BRDF encoder network size: ', brdf_encoder_dims)
        print('BRDF decoder network size: ', brdf_decoder_dims)

        brdf_encoder_layer = []
        dim = self.brdf_input_dim
        for i in range(len(brdf_encoder_dims)):
            brdf_encoder_layer.append(nn.Linear(dim, brdf_encoder_dims[i]))
            brdf_encoder_layer.append(self.actv_fn) 
            dim = brdf_encoder_dims[i]
        brdf_encoder_layer.append(nn.Linear(dim, self.latent_dim))
        self.brdf_encoder_layer = nn.Sequential(*brdf_encoder_layer)
        
        brdf_decoder_layer = []
        dim = self.latent_dim
        for i in range(len(brdf_decoder_dims)):
            brdf_decoder_layer.append(nn.Linear(dim, brdf_decoder_dims[i]))
            brdf_decoder_layer.append(self.actv_fn) 
            dim = brdf_decoder_dims[i]
        output_dim = 4
        brdf_decoder_layer.append(nn.Linear(dim, output_dim))
        self.brdf_decoder_layer = nn.Sequential(*brdf_decoder_layer)
        if metallic_range[1] != 0:
            self.brdf_matallic_layer = nn.Linear(dim, 1) # matal
        else:
            self.brdf_matallic_layer = None
        
        ################### light SGs ####################
        print('Number of Light SG: ', self.numLgtSGs)

        # by using normal distribution, the lobes are uniformly distributed on a sphere at initialization
        self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 7), requires_grad=True)   # [M, 7]; lobe + lambda + mu
        self.lgtSGs.data[:, -2:] = self.lgtSGs.data[:, -3:-2].expand((-1, 2))

        # make sure lambda is not too close to zero
        self.lgtSGs.data[:, 3:4] = 10. + torch.abs(self.lgtSGs.data[:, 3:4] * 20.)
        # init envmap energy
        energy = compute_energy(self.lgtSGs.data)
        self.lgtSGs.data[:, 4:] = torch.abs(self.lgtSGs.data[:, 4:]) / torch.sum(energy, dim=0, keepdim=True) * 2. * np.pi * 0.8
        energy = compute_energy(self.lgtSGs.data)
        # print('init envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())

        # deterministicly initialize lobes
        lobes = fibonacci_sphere(self.numLgtSGs//2).astype(np.float32)
        self.lgtSGs.data[:self.numLgtSGs//2, :3] = torch.from_numpy(lobes)
        self.lgtSGs.data[self.numLgtSGs//2:, :3] = torch.from_numpy(lobes)
        
        # check if lobes are in upper hemisphere
        self.upper_hemi = upper_hemi
        if self.upper_hemi:
            print('Restricting lobes to upper hemisphere!')
            self.restrict_lobes_upper = lambda lgtSGs: torch.cat((lgtSGs[..., :1], torch.abs(lgtSGs[..., 1:2]), lgtSGs[..., 2:]), dim=-1)
            # limit lobes to upper hemisphere
            self.lgtSGs.data = self.restrict_lobes_upper(self.lgtSGs.data) 

    def forward(self, points):
        if self.brdf_embed_fn is not None:
            points = self.brdf_embed_fn(points) 

        x = points 

        # encode
        for layer in self.brdf_encoder_layer:
            x = layer(x) 
        brdf_lc = torch.sigmoid(x)

        # decode
        y = brdf_lc
        for layer in self.brdf_decoder_layer[0:-1]:
            y = layer(y) 
        brdf_feat = y
        brdf = torch.sigmoid(self.brdf_decoder_layer[-1](brdf_feat))

        # parse feature
        roughness = brdf[..., 3:4] * 0.9 + 0.09 # [0.09, 0.99]
        roughness = torch.clip(roughness, self.roughness_range[0], self.roughness_range[1])
        if self.brdf_matallic_layer is None:
            metallic = torch.zeros_like(roughness).cuda()
        else:
            metallic = torch.clip(torch.sigmoid(self.brdf_matallic_layer(brdf_feat)), self.metallic_range[0], self.metallic_range[1])  # [0, 1]
        diffuse_albedo = torch.clip(brdf[..., :3], self.albedo_range[0], self.albedo_range[1])
        specular = torch.clip((1 - metallic) * self.default_specular_albedo + metallic * diffuse_albedo, 0.0, 1.0)

        rand_lc = brdf_lc + torch.randn(brdf_lc.shape).cuda() * 0.01
        random_xi_brdf = torch.sigmoid(self.brdf_decoder_layer(rand_lc))
        random_xi_roughness = random_xi_brdf[..., 3:4] * 0.9 + 0.09
        random_xi_roughness = torch.clip(random_xi_roughness, self.roughness_range[0], self.roughness_range[1])
        random_xi_specular = torch.zeros_like(random_xi_roughness).cuda()
        random_xi_diffuse = random_xi_brdf[..., :3]

        lgtSGs = self.lgtSGs
        if self.upper_hemi:
            # limit lobes to upper hemisphere
            lgtSGs = self.restrict_lobes_upper(lgtSGs)

        ret = dict([
            ('sg_lgtSGs', lgtSGs),
            ('sg_specular_reflectance', specular),
            ('sg_roughness', roughness),
            ('sg_metallic', metallic),
            ('sg_diffuse_albedo', diffuse_albedo),
            ('random_xi_roughness', random_xi_roughness),
            ('random_xi_diffuse_albedo', random_xi_diffuse),
            ('random_xi_specular_reflectance', random_xi_specular)
        ])
        return ret

    def get_light(self, is_eval=True):
        if is_eval:
            lgtSGs = self.lgtSGs.clone().detach()
        else:
            lgtSGs = self.lgtSGs
        # limit lobes to upper hemisphere
        if self.upper_hemi:
            lgtSGs = self.restrict_lobes_upper(lgtSGs)
        return lgtSGs

    def load_light(self, path):
        sg_path = os.path.join(path, 'sg_128.npy')
        device = self.lgtSGs.data.device
        load_sgs = torch.from_numpy(np.load(sg_path)).to(device)
        self.lgtSGs.data = load_sgs

        energy = compute_energy(self.lgtSGs.data)
        print('loaded envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())

        envmap_path = path + '.exr'
        envmap = np.float32(imageio.imread(envmap_path)[:, :, :3])
        self.envmap = torch.from_numpy(envmap).to(device)
    
    def load_exr_light(self, path):  
        envmap = np.float32(imageio.imread(path)[:, :, :3])
        envmap = torch.from_numpy(envmap).cuda()
        return envmap


def compute_envmap(lgtSGs, H, W, upper_hemi=False):
    # same convetion as blender    
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi/2., H), 
                                     torch.linspace(1.0 * np.pi, -1.0 * np.pi, W)], indexing='ij')
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), 
                                     torch.linspace(1.0 * np.pi, -1.0 * np.pi, W)], indexing='ij')
    viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), 
                            torch.sin(theta) * torch.sin(phi), 
                            torch.cos(phi)], dim=-1)    # [H, W, 3]
                            
    rgb = render_envmap_sg(lgtSGs, viewdirs)
    envmap = rgb.reshape((H, W, 3))
    return envmap


def render_envmap_sg(lgtSGs, viewdirs):
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]

    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([1,] * len(dots_sh) + [M, 7]).expand(dots_sh + [M, 7])
    
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True))
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:]) 
    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas *  (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]
    return rgb


def render_envmap(envmap, viewdirs):
    H, W = envmap.shape[:2]
    envmap = envmap.permute(2, 0, 1).unsqueeze(0)

    phi = torch.arccos(viewdirs[:, 2]).reshape(-1) - TINY_NUMBER
    theta = torch.atan2(viewdirs[:, 1], viewdirs[:, 0]).reshape(-1)

    # normalize to [-1, 1]
    query_y = (phi / np.pi) * 2 - 1
    query_x = - theta / np.pi
    grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)
    
    rgb = F.grid_sample(envmap, grid, align_corners=True)
    rgb = rgb.squeeze().permute(1, 0)
    return rgb

def hemisphere_int(lambda_val, cos_beta):
    lambda_val = lambda_val + TINY_NUMBER
    
    inv_lambda_val = 1. / lambda_val
    t = torch.sqrt(lambda_val) * (1.6988 + 10.8438 * inv_lambda_val) / (
                1. + 6.2201 * inv_lambda_val + 10.2415 * inv_lambda_val * inv_lambda_val)

    ### note: for numeric stability
    inv_a = torch.exp(-t)
    mask = (cos_beta >= 0).float()
    inv_b = torch.exp(-t * torch.clamp(cos_beta, min=0.))
    s1 = (1. - inv_a * inv_b) / (1. - inv_a + inv_b - inv_a * inv_b)
    b = torch.exp(t * torch.clamp(cos_beta, max=0.))
    s2 = (b - inv_a) / ((1. - inv_a) * (b + 1.))
    s = mask * s1 + (1. - mask) * s2

    A_b = 2. * np.pi / lambda_val * (torch.exp(-lambda_val) - torch.exp(-2. * lambda_val))
    A_u = 2. * np.pi / lambda_val * (1. - torch.exp(-lambda_val))

    return A_b * (1. - s) + A_u * s


def lambda_trick(lobe1, lambda1, mu1, lobe2, lambda2, mu2):
    # assume lambda1 << lambda2
    ratio = lambda1 / lambda2

    # for insurance
    lobe1 = norm_axis(lobe1)
    lobe2 = norm_axis(lobe2)
    dot = torch.sum(lobe1 * lobe2, dim=-1, keepdim=True)
    tmp = torch.sqrt(ratio * ratio + 1. + 2. * ratio * dot)
    tmp = torch.min(tmp, ratio + 1.)

    lambda3 = lambda2 * tmp
    lambda1_over_lambda3 = ratio / tmp
    lambda2_over_lambda3 = 1. / tmp
    diff = lambda2 * (tmp - ratio - 1.)

    final_lobes = lambda1_over_lambda3 * lobe1 + lambda2_over_lambda3 * lobe2
    final_lambdas = lambda3
    final_mus = mu1 * mu2 * torch.exp(diff)

    return final_lobes, final_lambdas, final_mus


def norm_axis(x):
    return x / (torch.norm(x, dim=-1, keepdim=True) + TINY_NUMBER)


def get_diffuse_visibility(points, normals, VisModel, lgtSGLobes, lgtSGLambdas, nsamp=8, batch_size = 2000,
                           uniform_sample=False, use_gt_vis=False):
    ########################################
    # sample dirs according to the light SG
    ########################################

    """
    points: [n_points, 3] 
    lgtSGLobes: [n_lobe, 3]
    lgtSGLambdas: [n_lobe, 1]
    """

    n_lobe = lgtSGLobes.shape[0] 
    n_points = points.shape[0]
    # light_dirs = lgtSGLobes.clone().detach().unsqueeze(-2)
    light_dirs = lgtSGLobes.unsqueeze(-2)
    lgtSGLambdas = lgtSGLambdas.clone().detach().unsqueeze(-2)

    # add samples from SG lobes
    z_axis = torch.zeros_like(light_dirs).cuda()
    z_axis[:, :, 2] = 1

    light_dirs = norm_axis(light_dirs) #[num_lobes, 1, 3]
    U = norm_axis(torch.cross(z_axis, light_dirs))
    V = norm_axis(torch.cross(light_dirs, U))
    # r_phi depends on the sg sharpness
    sharpness = lgtSGLambdas[:, :, 0]
    sg_range = torch.zeros_like(sharpness).cuda()
    sg_range[:, :] = sharpness.min()
    r_phi_range = torch.arccos((-1.95 * sg_range) / sharpness + 1)

    if uniform_sample:
        sample_step = ceil(nsamp ** 0.5)
        nsamp = sample_step ** 2
        r_theta = torch.linspace(0,1,steps=sample_step).cuda() * 2 * np.pi
        r_phi = torch.linspace(0,1,steps=sample_step).cuda() 
        r_theta, r_phi = torch.meshgrid(r_theta, r_phi, indexing='ij')
        r_theta = r_theta.reshape(1, -1).expand(n_lobe, -1)
        r_phi = r_phi.reshape(1, -1).expand(n_lobe, -1) * r_phi_range
    else:
        r_theta = torch.rand(n_lobe, nsamp).cuda() * 2 * np.pi
        r_phi = torch.rand(n_lobe, nsamp).cuda() * r_phi_range

    U = U.expand(-1, nsamp, -1)
    V = V.expand(-1, nsamp, -1)
    r_theta_expand = r_theta.unsqueeze(-1).expand(-1, -1, 3)
    r_phi_expand = r_phi.unsqueeze(-1).expand(-1, -1, 3)

    sample_dir = U * torch.cos(r_theta_expand) * torch.sin(r_phi_expand) \
                + V * torch.sin(r_theta_expand) * torch.sin(r_phi_expand) \
                + light_dirs * torch.cos(r_phi_expand) # [num_lobe, num_sample, 3]
    
    ########################################
    # visibility
    ########################################

    if use_gt_vis:

        sample_dir = sample_dir.reshape(-1, 3)
        input_dir = sample_dir.unsqueeze(0).expand(n_points, -1, 3)
        input_p = points.unsqueeze(1).expand(-1, n_lobe * nsamp, 3)
        normals = normals.unsqueeze(1).expand(-1, n_lobe * nsamp, 3)
        cos_mask = torch.sum(normals * input_dir, dim=-1) > TINY_NUMBER # vis = 0 if cos(n, w_i) < 0
        n_mask_dir = input_p[cos_mask].shape[0]

        pred_vis = torch.zeros(n_mask_dir).bool().cuda()
        with torch.no_grad():
            for i, indx in enumerate(torch.split(torch.arange(n_mask_dir).cuda(), batch_size, dim=0)):
                pred_vis[indx] = VisModel(input_p[cos_mask][indx], input_dir[cos_mask][indx])
        
        # pred_vis = F.softmax(pred_vis, dim=-1)[..., 1]  # [n_points * n_lobe * nsamp] 
        pred_vis = (~pred_vis).float()
         
    else:

        sample_dir = sample_dir.reshape(-1, 3)
        input_dir = sample_dir.unsqueeze(0).expand(n_points, -1, 3)
        input_p = points.unsqueeze(1).expand(-1, n_lobe * nsamp, 3)
        normals = normals.unsqueeze(1).expand(-1, n_lobe * nsamp, 3)
        cos_mask = torch.sum(normals * input_dir, dim=-1) > TINY_NUMBER # vis = 0 if cos(n, w_i) < 0
        n_mask_dir = input_p[cos_mask].shape[0]

        pred_vis = torch.zeros(n_mask_dir, 2).cuda()
        with torch.no_grad():
            for i, indx in enumerate(torch.split(torch.arange(n_mask_dir).cuda(), batch_size, dim=0)):
                pred_vis[indx] = VisModel(input_p[cos_mask][indx], input_dir[cos_mask][indx])

        pred_vis = F.softmax(pred_vis, dim=-1)[..., 1]  # [n_points * n_lobe * nsamp]
        # pred_vis = torch.max(pred_vis, dim=-1)[1].float()
        pred_vis = pred_vis.float()

        vis = torch.zeros(n_points, n_lobe * nsamp).cuda()  
        vis[cos_mask] = pred_vis 
        vis = vis.reshape(n_points, n_lobe, nsamp).permute(1, 2, 0)

    sample_dir = sample_dir.reshape(-1, nsamp, 3)

    # sample_dir [n_lobe, n_sample, 3]
    # light_dirs [n_lobe, 1, 3]
    # lgtSGLambdas [n_lobe, 1, 1]
    weight_vis = F.softmax(lgtSGLambdas * (torch.sum(sample_dir * light_dirs, dim=-1, keepdim=True) - 1.), dim=1)
    vis = torch.sum(vis * weight_vis, dim=1) 

    # for debugging
    if torch.isnan(vis).sum() > 0:
        import ipdb; ipdb.set_trace()
        
    return vis

def get_specular_visibility(points, normals, viewdirs, VisModel, lgtSGLobes, lgtSGLambdas, nsamp=24, batch_size = 100000,
                            use_gt_vis=False): 
    ########################################
    # sample dirs according to the BRDF SG
    ########################################

    # light_dirs = lgtSGLobes.clone().detach().unsqueeze(-2)
    light_dirs = lgtSGLobes.unsqueeze(-2)
    lgtSGLambdas = lgtSGLambdas.clone().detach().unsqueeze(-2)
    
    n_points = points.shape[0] 

    n_dot_v = torch.sum(normals * viewdirs, dim=-1, keepdim=True)
    n_dot_v = torch.clamp(n_dot_v, min=0.)
    ref_dir = -viewdirs + 2 * n_dot_v * normals
    ref_dir = ref_dir.unsqueeze(1)
    
    # add samples from BRDF SG lobes
    z_axis = torch.zeros_like(ref_dir).cuda()
    z_axis[:, :, 2] = 1

    U = norm_axis(torch.cross(z_axis, ref_dir))
    V = norm_axis(torch.cross(ref_dir, U))
    # r_phi depends on the sg sharpness
    sharpness = lgtSGLambdas[:, :, 0]
    sharpness = torch.clip(sharpness, min=0.1, max=50)
    sg_range = torch.zeros_like(sharpness).cuda()
    sg_range[:, :] = sharpness.min()
    r_phi_range = torch.arccos((-1.90 * sg_range) / sharpness + 1)
  
    r_theta = torch.rand(ref_dir.shape[0], nsamp).cuda() * 2 * np.pi
    r_phi = torch.rand(ref_dir.shape[0], nsamp).cuda() * r_phi_range

    U = U.expand(-1, nsamp, -1)
    V = V.expand(-1, nsamp, -1)
    r_theta = r_theta.unsqueeze(-1).expand(-1, -1, 3)
    r_phi = r_phi.unsqueeze(-1).expand(-1, -1, 3)

    sample_dir = U * torch.cos(r_theta) * torch.sin(r_phi) \
                + V * torch.sin(r_theta) * torch.sin(r_phi) \
                + ref_dir * torch.cos(r_phi)

    if use_gt_vis:

        input_p = points.unsqueeze(1).expand(-1, nsamp, 3)
        input_dir = sample_dir
        normals = normals.unsqueeze(1).expand(-1, nsamp, 3) 

        cos_mask = torch.sum(normals * input_dir, dim=-1) > TINY_NUMBER
        cos_mask = cos_mask.detach()
        n_mask_dir = input_p[cos_mask].shape[0] 

        pred_vis = torch.zeros(n_mask_dir).bool().cuda()

        with torch.no_grad(): 
            for i, indx in enumerate(torch.split(torch.arange(n_mask_dir).cuda(), batch_size, dim=0)):
                pred_vis[indx] = VisModel(input_p[cos_mask][indx], input_dir[cos_mask][indx]) 
        
        pred_vis = (~pred_vis).float() 
    
    else:
    
        input_p = points.unsqueeze(1).expand(-1, nsamp, 3)
        input_dir = sample_dir
        normals = normals.unsqueeze(1).expand(-1, nsamp, 3) 

        cos_mask = torch.sum(normals * input_dir, dim=-1) > TINY_NUMBER
        cos_mask = cos_mask.detach()
        n_mask_dir = input_p[cos_mask].shape[0] 

        pred_vis = torch.zeros(n_mask_dir, 2).cuda()

        with torch.no_grad(): 
            for i, indx in enumerate(torch.split(torch.arange(n_mask_dir).cuda(), batch_size, dim=0)):
                pred_vis[indx] = VisModel(input_p[cos_mask][indx], input_dir[cos_mask][indx]) 
                    
        pred_vis = F.softmax(pred_vis, dim=-1)[..., 1]
        # pred_vis = torch.max(pred_vis, dim=-1)[1].float()

        vis = torch.zeros(points.shape[0], nsamp).cuda() 
        vis[cos_mask] = pred_vis  
 
    weight_vis = F.softmax(sharpness * (torch.sum(sample_dir * light_dirs, dim=-1) - 1.), dim=-1) # []

    inf_idx = torch.isinf(torch.sum(weight_vis, dim=-1))
    inf_sample = weight_vis[inf_idx]

    if inf_sample.sum() > 0:
        print('Nan Error')
        sys.exit(0)
    
    final_vis = torch.sum(vis * weight_vis, dim=-1)  
    # for debugging
    if torch.isnan(vis).sum() > 0:
        import ipdb; ipdb.set_trace()

    return final_vis

def identity_vis(x,y):
    N = x.shape[0]
    vis = torch.cat([torch.zeros([N, 1]), torch.ones([N, 1])], dim=1).cuda()
    return vis

def render_with_all_sg(points, normal, viewdirs, lgtSGs, 
                       specular_reflectance, roughness,
                       metallic, diffuse_albedo,
                       indir_lgtSGs=None, VisModel=None,
                       use_mesh_vis=False, forbid_vis=False):
    
    if forbid_vis:
        VisModel = identity_vis
        warnings.warn('### Use no visibility field. ###')

    M = lgtSGs.shape[0]
    n_points = normal.shape[0]

    # direct light
    lgtSGs_input = lgtSGs.unsqueeze(0).expand([n_points, M, 7])  # [dots_shape, M, 7]
    ret = render_with_sg(points, normal, viewdirs, lgtSGs_input, 
                         specular_reflectance, roughness,
                         metallic, diffuse_albedo,  
                         render_indirect=False,
                         VisModel=VisModel, 
                         use_mesh_vis=use_mesh_vis)

    # indirct light
    indir_rgb = torch.zeros_like(points).cuda()
    if indir_lgtSGs is not None:  
        indir_rgb = render_with_sg(points, normal, viewdirs, indir_lgtSGs, 
                                   specular_reflectance, roughness,
                                   metallic, diffuse_albedo,
                                   render_indirect=True,
                                   VisModel=VisModel, 
                                   use_mesh_vis=use_mesh_vis)['sg_rgb']
    ret.update({'indir_rgb': indir_rgb})

    return ret


#######################################################################################################
# below is the SG renderer
#######################################################################################################
def render_with_sg(points, normal, viewdirs, 
                   lgtSGs, specular_reflectance, roughness,
                   metallic, diffuse_albedo,
                   VisModel=None, render_indirect=False, 
                   use_mesh_vis=False):
    '''
    :param points: [batch_size, 3]
    :param normal: [batch_size, 3]; ----> camera; must have unit norm
    :param viewdirs: [batch_size, 3]; ----> camera; must have unit norm
    :param lgtSGs: [batch_size, M, 7] or [obj_num, M, 7]
    :param specular_reflectance: [1, 1] or [batch_size, 3]; 
    :param roughness: [batch_size, 1]; values must be positive
    :param diffuse_albedo: [batch_size, 3]; values must lie in [0,1]
    '''

    M = lgtSGs.shape[1]
    dots_shape = list(normal.shape[:-1])

    ########################################
    # light
    ########################################

    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER) # lobe axis direction
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4]) # sharpness 增加该值时,波瓣会变得更纤细,也就意味着越是远离波瓣轴衰减的越快
    
    origin_lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values
    
    ########################################
    # specular color
    ########################################
    normal = normal.unsqueeze(-2).expand(dots_shape + [M, 3])  # [dots_shape, M, 3]
    viewdirs = viewdirs.unsqueeze(-2).expand(dots_shape + [M, 3]).detach()  # [dots_shape, M, 3]
    
    # NDF
    brdfSGLobes = normal  # use normal as the brdf SG lobes
    inv_roughness_pow4 = 2. / (roughness ** 4)  # [dots_shape, 1]
    brdfSGLambdas = inv_roughness_pow4.unsqueeze(1).expand(dots_shape + [M, 1])
    mu_val = (inv_roughness_pow4 / np.pi).expand(dots_shape + [3])  # [dots_shape, 1] ---> [dots_shape, 3]
    brdfSGMus = mu_val.unsqueeze(1).expand(dots_shape + [M, 3])

    # perform spherical warping
    v_dot_lobe = torch.sum(brdfSGLobes * viewdirs, dim=-1, keepdim=True)
    ### note: for numeric stability
    v_dot_lobe = torch.clamp(v_dot_lobe, min=0.)
    warpBrdfSGLobes = 2 * v_dot_lobe * brdfSGLobes - viewdirs
    warpBrdfSGLobes = warpBrdfSGLobes / (torch.norm(warpBrdfSGLobes, dim=-1, keepdim=True) + TINY_NUMBER)
    warpBrdfSGLambdas = brdfSGLambdas / (4 * v_dot_lobe + TINY_NUMBER)
    warpBrdfSGMus = brdfSGMus  # [..., M, 3]

    new_half = warpBrdfSGLobes + viewdirs
    new_half = new_half / (torch.norm(new_half, dim=-1, keepdim=True) + TINY_NUMBER)
    v_dot_h = torch.sum(viewdirs * new_half, dim=-1, keepdim=True)
    ### note: for numeric stability+
    v_dot_h = torch.clamp(v_dot_h, min=0.)

    specular_reflectance = specular_reflectance.unsqueeze(1).expand(dots_shape + [M, 3])
    F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)

    dot1 = torch.sum(warpBrdfSGLobes * normal, dim=-1, keepdim=True)  # equals <o, n>
    ### note: for numeric stability
    dot1 = torch.clamp(dot1, min=0.)
    dot2 = torch.sum(viewdirs * normal, dim=-1, keepdim=True)  # equals <o, n>
    ### note: for numeric stability
    dot2 = torch.clamp(dot2, min=0.)
    k = (roughness + 1.) * (roughness + 1.) / 8.
    k = k.unsqueeze(1).expand(dots_shape + [M, 1])
    G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
    G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
    G = G1 * G2

    Moi = F * G / (4 * dot1 * dot2 + TINY_NUMBER)
    warpBrdfSGMus = warpBrdfSGMus * Moi 
    
    ########################################
    # calculate visibility
    ########################################

    # 原始情况,直接光
    light_vis = None
    if not render_indirect:

        nsamp = 25 if not use_mesh_vis else 4
        batch_size = 100000 if not use_mesh_vis else 10000

        # light SG visibility
        light_vis = get_diffuse_visibility(points, normal[:, 0, :], VisModel, lgtSGLobes[0], lgtSGLambdas[0],
                                           nsamp=nsamp, batch_size=batch_size, use_gt_vis=False, uniform_sample=True)
        light_vis = light_vis.permute(1, 0).unsqueeze(-1).expand(dots_shape +[M, 3])

        # BRDF SG visibility
        brdf_vis = get_specular_visibility(points, normal[:, 0, :], viewdirs[:, 0, :], VisModel, warpBrdfSGLobes[:, 0], warpBrdfSGLambdas[:, 0],
                                           nsamp=nsamp, batch_size=batch_size, use_gt_vis=False)
        brdf_vis = brdf_vis.unsqueeze(-1).unsqueeze(-1).expand(dots_shape + [M, 3])

        lgtSGMus = origin_lgtSGMus * brdf_vis
        
    
    # 原始情况,间接光
    else:
        lgtSGMus = origin_lgtSGMus
    
    
    ########################################
    # per-point hemisphere integral of envmap for specular light
    ########################################

    # multiply with light sg
    final_lobes, final_lambdas, final_mus = lambda_trick(lgtSGLobes, lgtSGLambdas, lgtSGMus, warpBrdfSGLobes, warpBrdfSGLambdas, warpBrdfSGMus)

    # now multiply with clamped cosine, and perform hemisphere integral
    mu_cos = 32.7080
    lambda_cos = 0.0315
    alpha_cos = 31.7003
    lobe_prime, lambda_prime, mu_prime = lambda_trick(normal, lambda_cos, mu_cos, final_lobes, final_lambdas, final_mus)

    dot1 = torch.sum(lobe_prime * normal, dim=-1, keepdim=True)
    dot2 = torch.sum(final_lobes * normal, dim=-1, keepdim=True)
    # [..., M, K, 3]
    specular_rgb = mu_prime * hemisphere_int(lambda_prime, dot1) - final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)
    specular_rgb = torch.sum(specular_rgb, dim=-2)
    specular_rgb = torch.clamp(specular_rgb, min=0.) 


    ########################################
    # per-point hemisphere integral of envmap for diffuse light
    ########################################
    # diffuse visibility
    if not render_indirect:
        lgtSGMus = origin_lgtSGMus * light_vis
    else:
        lgtSGMus = origin_lgtSGMus  
    # diffuse =  ((1.0 - metallic) *diffuse_albedo / np.pi).unsqueeze(-2).expand(dots_shape + [M, 3])
    diffuse =  ( diffuse_albedo / np.pi).unsqueeze(-2).expand(dots_shape + [M, 3])
    
    # multiply with light sg
    final_lobes = lgtSGLobes
    final_lambdas = lgtSGLambdas
    final_mus = lgtSGMus * diffuse

    # now multiply with clamped cosine, and perform hemisphere integral
    lobe_prime, lambda_prime, mu_prime = lambda_trick(normal, lambda_cos, mu_cos, final_lobes, final_lambdas, final_mus)

    dot1 = torch.sum(lobe_prime * normal, dim=-1, keepdim=True)
    dot2 = torch.sum(final_lobes * normal, dim=-1, keepdim=True)
    diffuse_rgb = mu_prime * hemisphere_int(lambda_prime, dot1) - final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)
    diffuse_rgb = diffuse_rgb.sum(dim=-2)
    diffuse_rgb = torch.clamp(diffuse_rgb, min=0.)

    # combine diffue and specular rgb
    # no clip, beacuse we train with HDR image
    rgb = specular_rgb + diffuse_rgb
    ret = {'sg_rgb': rgb,
           'sg_specular_rgb': specular_rgb,
           'sg_diffuse_rgb': diffuse_rgb,
           'light_vis':light_vis}

    return ret
