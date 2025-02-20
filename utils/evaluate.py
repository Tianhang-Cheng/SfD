import torch
import os
import numpy as np

from utils import rend_util

from skimage.metrics import structural_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

PI = np.pi

tonemap_img = lambda x: torch.pow(x, 1./2.2) 
clip_img = lambda x: torch.clamp(x, min=0., max=1.)
norm_vector = lambda x: x / (torch.norm(x, dim=-1, keepdim=True) + 1e-6)

mse2psnr = lambda x: -10. * np.log(x + 1e-8) / np.log(10.)  
ssim = lambda x, y: structural_similarity(x, y, data_range=max(np.max(x), np.max(y))-min(np.min(x), np.min(y)), channel_axis=2, multichannel=True)
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()
mse = lambda x,y: torch.mean(torch.square((x-y)))

def transform_image(x):
    return tonemap_img(clip_img(x)) 
 
@torch.no_grad()
def evaluate_all(pred_rgb, object_mask, network_mask, rgb_gt, path, iters, img_res, use_union_mask,
                 postfix='', prefix='', parent_dir=True, align_channel=False, **kwargs):
    if use_union_mask:
        mask = object_mask & network_mask
        mask_name = 'union_mask'
    else:
        mask = object_mask
        mask_name = 'obj_mask'
    
    pred_rgb = transform_image(pred_rgb) 
    rgb_gt = transform_image(rgb_gt) 
    
    # psnr
    rgb_mask = pred_rgb[mask]
    rgb_gt_mask = rgb_gt[mask]

    if align_channel:
        if 'align_scale' in kwargs.keys():
            align_scale = kwargs['align_scale']
        else:
            align_scale = torch.sum(rgb_gt_mask, dim=0) / torch.sum(rgb_mask, dim=0) # [3] 
        pred_rgb = clip_img(pred_rgb * align_scale) # for draw
        rgb_mask = clip_img(rgb_mask * align_scale) # for loss
    else:
        align_scale = torch.tensor([1,1,1]).float().cuda()

    mse_loss = torch.nn.functional.mse_loss(rgb_mask, rgb_gt_mask).cpu().numpy()
    psnr = mse2psnr(mse_loss) 

    # ssim, set invalid area to 1
    size = img_res[0]
    mask_reshape = mask.clone().reshape([size,size])
    rgb_reshape = pred_rgb.clone().reshape([size,size,3])
    rgb_reshape[~mask_reshape] = 1.0
    rgb_gt_reshape = rgb_gt.clone().reshape([size,size,3])
    rgb_gt_reshape[~mask_reshape] = 1.0
    ssim_loss = ssim(rgb_reshape.cpu().numpy(), rgb_gt_reshape.cpu().numpy())

    # lpips
    rgb_permute = rgb_reshape.permute(2,0,1).unsqueeze(0).cuda()
    rgb_gt_permute = rgb_gt_reshape.permute(2,0,1).unsqueeze(0).cuda()
    lpips_loss = lpips(rgb_permute * 2 - 1, rgb_gt_permute * 2 - 1).detach().cpu() # should scale to [-1,1]

    # normal 
    evaluate_normal = False
    if 'normals' in kwargs.keys():
        evaluate_normal = True
        normal_gt = norm_vector(clip_img(kwargs['normals_gt']))
        normal_pred = norm_vector(clip_img(rend_util.w2c_normal(kwargs['normals'][None], kwargs['pose']) * 0.5 + 0.5))[0]
        diff_angle = torch.sum(normal_gt*normal_pred, dim=-1)
        diff_angle = torch.acos(diff_angle) / PI * 180.0
        normal_abs_loss = torch.mean(diff_angle[mask]).cpu().numpy()
    
    # depth 
    # evaluate_depth = False
    # if 'points' in kwargs.keys() and kwargs['points'] is not None:
    #     evaluate_depth = True
    #     scale = kwargs['scale']
    #     pose = kwargs['pose']  
    #     points = kwargs['points'].reshape(1,-1,3) 
    #     depth = torch.ones_like(mask).cuda().float() 
    #     if mask.sum() > 0:
    #         depth[mask] = rend_util.get_depth(points, pose, scale=scale).reshape(-1)[mask]
    #         depth_mse_loss = mse(kwargs['depth_gt'][mask], depth[mask]).cpu().numpy()
    #     else:
    #         depth_mse_loss = -1.0
    
    # roughness
    evaluate_roughness = False
    if 'roughness_gt' in kwargs.keys() and kwargs['roughness_gt'] is not None:
        evaluate_roughness = True
        rough_gt = transform_image(kwargs['roughness_gt'])
        rough_pred = transform_image(kwargs['roughness'])
        rough_mse_loss = mse(rough_gt[mask], rough_pred[mask]).cpu().numpy()

    if parent_dir:
        save_path = os.path.dirname(path)
    else:
        save_path = path

    write_path = os.path.join(save_path, prefix+'_psnr_'+mask_name+postfix+'.txt')
    with open(write_path, "a+")as f:
        f.write('iters [{0}]:psnr = {1} \n'.format(iters, psnr))

    write_path = os.path.join(save_path, prefix+'_ssim_'+mask_name+postfix+'.txt')
    with open(write_path, "a+")as f:
        f.write('iters [{0}]:ssim = {1} \n'.format(iters, ssim_loss))
    
    write_path = os.path.join(save_path, prefix+'_lpips_'+mask_name+postfix+'.txt')
    with open(write_path, "a+")as f:
        f.write('iters [{0}]:lpips = {1} \n'.format(iters, lpips_loss))

    if align_channel:
        write_path = os.path.join(save_path, prefix+'_channel_scales_'+mask_name+postfix+'.txt')
        with open(write_path, "a+")as f:
            f.write('iters [{0}]:scale = {1} \n'.format(iters, align_scale.detach().cpu().numpy()))
    
    result_dict = {'psnr': psnr,
                   'ssim': ssim_loss,
                   'lpips': lpips_loss,
                   'align_scale': align_scale}

    # if evaluate_depth:

    #     write_path = os.path.join(save_path, 'depth_mse_'+mask_name+postfix+'.txt')
    #     with open(write_path, "a+")as f:
    #         f.write('iters [{0}]:depth_mse = {1} \n'.format(iters, depth_mse_loss))

    #     result_dict.update({'depth_mse': depth_mse_loss})
    
    if evaluate_normal:

        write_path = os.path.join(save_path, 'normal_abs_'+mask_name+postfix+'.txt')
        with open(write_path, "a+")as f:
            f.write('iters [{0}]:normal_abs = {1}° \n'.format(iters, normal_abs_loss))

        result_dict.update({'normal_abs': normal_abs_loss})

    if evaluate_roughness:

        write_path = os.path.join(save_path, 'roughness_mse_'+mask_name+postfix+'.txt')
        with open(write_path, "a+")as f:
            f.write('iters [{0}]:roughness_mse = {1} \n'.format(iters, rough_mse_loss))

        result_dict.update({'roughness_mse': rough_mse_loss})
 
    # debug_path = os.path.join(save_path, 'debug_neus_psnr_union_mask.txt')
    # with open(debug_path, "a+")as f:
    #     f.write('min rgb = {0} \n'.format(rgb.min()))
    #     f.write('min rgb gt = {0} \n'.format(rgb_gt.min()))
    #     if torch.isnan(rgb).sum() > 0:
    #         f.write('rgb = {0} \n'.format(torch.isnan(rgb).sum()))
    #     if torch.isnan(tonemap_img(rgb)).sum() > 0:
    #         f.write('tonemap rgb = {0} \n'.format(torch.isnan(tonemap_img(rgb)).sum()))
    #     if torch.isnan(rgb_gt).sum() > 0:
    #         f.write('rgb_gt = {0} \n'.format(torch.isnan(rgb_gt).sum()))
    #     if torch.isnan(tonemap_img(rgb_gt)).sum() > 0:
    #         f.write('tonemap rgb_gt = {0} \n'.format(torch.isnan(tonemap_img(rgb_gt)).sum()))
    # exit(0)

    return result_dict