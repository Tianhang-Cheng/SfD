
import numpy as np
import torch
import torchvision
from PIL import Image
from utils import rend_util 
import imageio
import matplotlib.pyplot as plt

tonemap_img = lambda x: torch.pow(x, 1./2.2) 
clip_img = lambda x: torch.clamp(x, min=0., max=1.)
norm_vector = lambda x: x / (torch.norm(x, dim=-1, keepdim=True) + 1e-6)

def plot_class(pred_class, ground_true, object_mask, path, iters, img_res, same_obj_num, prefix=None):

    """
    pred_class: [1,H*W,1]
    ground_true: [1, H*W]
    mask: [H*W]
    """

    pred_class_plot = pred_class[..., 0] + 1
    pred_class_plot = clip_img(pred_class_plot / same_obj_num)   # [1,H*W]: [0,255]
    mask = ~object_mask[None]
    pred_class_plot[mask] = 0.

    ground_true = ground_true.cuda().float()
    ground_true = clip_img((ground_true) / ground_true.max()) 

    diff = torch.abs(pred_class_plot - ground_true)

    output_vs_gt = torch.cat((pred_class_plot * 255., ground_true * 255., diff * 255.), dim=0)
    output_vs_gt_plot = output_vs_gt.reshape(-1, img_res[0], img_res[1])

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=3).cpu().detach().numpy()


    tensor = tensor.astype(np.uint8)
    b,h,w = tensor.shape
    tensor = tensor.reshape(b*h,w)

    cmap = plt.get_cmap('viridis')
    result = cmap(tensor)
    img = Image.fromarray((result * 255).astype(np.uint8))
    if prefix is not None:
        print('saving classify img to {}/{}_classification_{}.png'.format(path, prefix, iters))
        img.save('{}/{}_classification_{}.png'.format(path, prefix, iters))
    else:
        print('saving classify img to {}/classification_{}.png'.format(path, iters))
        img.save('{}/classification_{}.png'.format(path, iters)) 

def plot_rgb(model_outputs, rgb_gt, path, iters, img_res, name):

    batch_size, num_samples, _ = rgb_gt.shape
 
    network_object_mask = model_outputs['network_object_mask'].reshape(batch_size, num_samples, 1).bool()
    obj_mask = model_outputs['object_mask'].reshape(batch_size, num_samples, 1) 
    union_mask = network_object_mask & obj_mask

    rgb = model_outputs['sg_rgb'].reshape(batch_size, num_samples, 3) 
    rgb[~union_mask[..., 0]] = 1.0
    rgb_gt[~obj_mask[...,0]] = 1.0
 
    rgb = clip_img(tonemap_img(rgb)) 
    rgb_gt = clip_img(tonemap_img(rgb_gt))

    output_vs_gt = torch.cat((rgb, rgb_gt), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    # import pdb
    # pdb.set_trace()

    x = (rgb.reshape([800,800,3]).detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(x)
    img.save('{0}/relight_{1}.png'.format(path,name))

    x = (rgb_gt.reshape([800,800,3]).detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(x)
    img.save('{0}/relight_gt.png'.format(path))

    print('Finish')

    # tensor = torchvision.utils.make_grid(output_vs_gt_plot,
    #                                      scale_each=False,
    #                                      normalize=False,
    #                                      nrow=1).cpu().detach().numpy()

    # tensor = tensor.transpose(1, 2, 0)
    # scale_factor = 255
    # tensor = (tensor * scale_factor).astype(np.uint8)

    # img = Image.fromarray(tensor)
    # print('saving render img to {0}/rendering_{1}.png'.format(path, iters))
    # img.save('{0}/rendering_{1}.png'.format(path, iters))


def plot_depth_maps(depth_maps, path, iters, img_res, save_exr=False):
    depth_maps_plot = lin2img(depth_maps, img_res)

    tensor = torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1),
                                         scale_each=True,
                                         normalize=True,
                                         nrow=1).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255

    tensor_draw = (tensor * scale_factor).astype(np.uint8)
    img = Image.fromarray(tensor_draw)
    img.save('{0}/depth_{1}.png'.format(path, iters))

    if save_exr:
        imageio.imwrite('{0}/depth_{1}.exr'.format(path, iters), tensor)
        exit(0)

def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])


def plot_neus(rgb_pred, normal_img, rgb_gt, normal_gt, object_mask, network_mask, path, iters, img_res,
              gt_seg=None, sdf_pred_seg=None, proxy_pred_seg=None, surface_points=None, pose=None, same_obj_num=0,
              use_pretrain_normal=False, pretrain_normal_gt=None):
    """
    all input size should be [H, W, 3] or [H, W]
    color space: linear and mapped(srgb)
    masked: mask by object mask and network mask
    """
    print('saving rendering results to {}'.format(path))

    # Mask
    object_mask_ = object_mask.unsqueeze(-1)
    network_mask_ = network_mask.unsqueeze(-1)

    # RGB
    rgb_gt_linear = rgb_gt.clone() # linear space
    rgb_gt_mapped = clip_img(tonemap_img(rgb_gt_linear))
    rgb_pred_mapped_full = clip_img(tonemap_img(rgb_pred))
    rgb_pred_mapped_masked = clip_img(tonemap_img(rgb_pred * object_mask_ * network_mask_))
    rgb_diff_mapped_full = clip_img(tonemap_img(torch.abs(rgb_gt_linear * object_mask_ - rgb_pred)))
    rgb_diff_mapped_masked = clip_img(tonemap_img(torch.abs(rgb_gt_linear - rgb_pred) * object_mask_ * network_mask_))

    x = (rgb_gt_mapped.detach().cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite('{}/rendering_gt_mapped.png'.format(path), x)

    x = (rgb_pred_mapped_full.detach().cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite('{}/rendering_pred_mapped_full_{}.png'.format(path, iters), x)

    x = (rgb_pred_mapped_masked.detach().cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite('{}/rendering_pred_mapped_masked_{}.png'.format(path, iters), x)

    x = (torch.cat([rgb_diff_mapped_full, rgb_diff_mapped_masked], dim=1).detach().cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite('{}/rendering_error_mapped_{}.png'.format(path, iters), x)

    # Normal
    normal = clip_img((normal_img + 1.0) / 2.0) 
    normal_masked = normal.clone()
    normal_masked[~network_mask] = 1.0
    normal_masked[~object_mask] = 1.0 

    if normal_gt is not None:
        normal_gt[~object_mask] = 1.0
        x = (normal_gt.detach().cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite('{}/normal_gt.png'.format(path), x)
        # plt.imshow((normal_gt).detach().cpu().numpy().reshape([800, 800, 3]) *255)
        # plt.imshow(x)

    if use_pretrain_normal and pretrain_normal_gt is not None:
        pretrain_normal_gt[~object_mask] = 1.0
        x = (pretrain_normal_gt.detach().cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite('{}/normal_pretrain.png'.format(path), x)

    x = (normal.detach().cpu().numpy() * 255).astype(np.uint8) 
    imageio.imwrite('{}/normal_full_{}.png'.format(path, iters), x)

    x = (normal_masked.detach().cpu().numpy() * 255).astype(np.uint8) 
    imageio.imwrite('{}/normal_masked_{}.png'.format(path, iters), x)   

    # Segmentation
    if sdf_pred_seg is not None: 
        pred_seg = sdf_pred_seg.reshape(1, img_res[0]*img_res[1], 1)
        gt_seg = gt_seg.reshape(1, img_res[0]*img_res[1])
        plot_class(pred_seg, gt_seg, (network_mask&object_mask).reshape(-1), path, iters, img_res, same_obj_num, prefix='sdf')
    if proxy_pred_seg is not None: 
        pred_seg = proxy_pred_seg.reshape(1, img_res[0]*img_res[1], 1)
        gt_seg = gt_seg.reshape(1, img_res[0]*img_res[1])
        plot_class(pred_seg, gt_seg, (network_mask&object_mask).reshape(-1), path, iters, img_res, same_obj_num, prefix='proxy')
    
    # Depth
    # if surface_points is not None:
    #     depth = torch.ones(batch_size * num_samples).cuda().float() 
    #     network_object_mask = network_mask & object_mask
    #     if network_object_mask.sum() > 0:
    #         depth_valid = rend_util.get_depth(surface_points, pose).reshape(-1)[network_object_mask]
    #         depth[network_object_mask] = depth_valid
    #         depth[~network_object_mask] = 0.98 * depth_valid.min()
    #     depth = depth.reshape(batch_size, num_samples, 1)

    #     # plot depth maps
    #     plot_depth_maps(depth, path, iters, img_res)

def plot_neus_vis(pred_vis, gt_vis, rgb_gt, path, iters, img_res):

    pred_vis = pred_vis.unsqueeze(-1).expand(-1, -1, 3)
    gt_vis = gt_vis.unsqueeze(-1).expand(-1, -1, 3)

    if rgb_gt is not None:
        rgb_gt = clip_img(tonemap_img(rgb_gt.cuda()))
        output_vs_gt = torch.stack((pred_vis, gt_vis, rgb_gt), dim=0).permute(0,3,1,2)
    else:
        output_vs_gt = torch.stack((pred_vis, gt_vis), dim=0).permute(0,3,1,2)

    tensor = torchvision.utils.make_grid(output_vs_gt,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=2).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    print('saving vis img to {0}/visibility_{1}.png'.format(path, iters))
    img.save('{0}/visibility_{1}.png'.format(path, iters))
 
    x = (pred_vis.detach().cpu().numpy() * 255).astype(np.uint8) 
    img = Image.fromarray(x)
    img.save('{0}/pred_visibility_{1}.png'.format(path, iters))

def plot_neus_mat(normal, normal_gt,
                  vis_shadow,
                  diffuse_albedo, diffuse_albedo_gt,
                  roughness, roughness_gt,
                  metallic, metallic_gt,
                  specular_rgb,
                  sg_rgb, rgb_gt,
                #   depth_gt,
                  pose,
                  points,
                  obj_mask, network_obj_mask,
                  path, iters, **kwargs):
    H, W, _ = rgb_gt.shape

    rgb_gt = rgb_gt.cuda() 
    ground_true_draw = clip_img(tonemap_img(rgb_gt))
    
    # rgb_diff1 = clip_img(tonemap_img(torch.abs(rgb_gt - sg_rgb) * (obj_mask)))
    # rgb_diff2 = clip_img(tonemap_img(torch.abs(rgb_gt - sg_rgb) * (obj_mask & network_obj_mask))) 

    pose = pose.reshape(1,4,4)

    normal = normal.reshape(1, H*W, 3)
    normal = rend_util.w2c_normal(normals=normal, pose=pose)
    normal = clip_img((normal + 1.) / 2.)
    normal = normal.reshape([H,W,3])

    empty =  torch.ones_like(rgb_gt).cuda()

    if normal_gt is None:
        normal_gt = empty
    if diffuse_albedo_gt is None:
        diffuse_albedo_gt = empty
    if roughness_gt is None:
        roughness_gt = empty
    if metallic is None:
        metallic = empty
    if metallic_gt is None:
        metallic_gt = empty

    if 'align_scale' in kwargs.keys():
        align_scale = kwargs['align_scale']
    else:
        align_scale = torch.tensor([1.0,1.0,1.0]).cuda()
    
    diffuse_albedo_align = diffuse_albedo * align_scale

    # import pdb
    # pdb.set_trace()

    specular_rgb = clip_img(tonemap_img(specular_rgb))
    sg_rgb = clip_img(tonemap_img(sg_rgb))
    diffuse_albedo = clip_img(tonemap_img(diffuse_albedo))
    diffuse_albedo_align = clip_img(tonemap_img(diffuse_albedo_align))
    diffuse_albedo_gt = clip_img(tonemap_img(diffuse_albedo_gt)) 

    union_mask = obj_mask & network_obj_mask

    rgb_gt[~obj_mask[...,0]] = 1.0
    sg_rgb[~union_mask[...,0]] = 1.0
    vis_shadow[~obj_mask[...,0]] = 1.0
    # depth_gt[~obj_mask[...,0]] = 1.0
    diffuse_albedo[~union_mask[...,0]] = 1.0
    diffuse_albedo_align[~union_mask[...,0]] = 1.0
    diffuse_albedo_gt[~obj_mask[...,0]] = 1.0
    roughness[~union_mask[...,0]] = 1.0
    # roughness_gt = tonemap_img(roughness_gt)
    roughness_gt[~obj_mask[...,0]] = 1.0
    metallic[~union_mask[...,0]] = 1.0
    metallic_gt[~obj_mask[...,0]] = 1.0
    # metallic_gt = tonemap_img(metallic_gt)

    # depth = torch.ones(H*W).cuda().float() 
    # if network_obj_mask.sum() > 0:
    #     temp_mask = network_obj_mask.flatten()
    #     depth_valid = rend_util.get_depth(points, pose, scale=1).reshape(-1)[temp_mask]
    #     depth[temp_mask] = depth_valid
    #     depth[~temp_mask] = 1.0
    # depth = depth.reshape(H,W, 1).expand(-1,-1,3)

    normal[~union_mask[...,0]] = 1.0
    normal_gt[~obj_mask[...,0]] = 1.0
    ground_true_draw[~obj_mask[...,0]] = 1.0

    #########################

    # import pdb
    # pdb.set_trace()
    
    name = 'our'

    x = (normal .detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(x)
    img.save('{0}/normal_{1}_{2}.png'.format(path,name,iters))

    x = (normal_gt.detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(x)
    img.save('{0}/normal_gt.png'.format(path))

    x = (diffuse_albedo.detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(x)
    img.save('{0}/diffuse_{1}_{2}.png'.format(path,name,iters))

    x = (diffuse_albedo_gt.detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(x)
    img.save('{0}/diffuse_gt.png'.format(path))

    x = (roughness.detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(x)
    img.save('{0}/rough_{1}_{2}.png'.format(path,name,iters)) 

    x = (roughness_gt.detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(x)
    img.save('{0}/rough_gt.png'.format(path))

    x = (metallic.detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(x)
    img.save('{0}/metal_{1}_{2}.png'.format(path,name,iters))

    x = (metallic_gt.detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(x)
    img.save('{0}/metal_gt.png'.format(path))

    x = (sg_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(x)
    img.save('{0}/rgb_{1}_{2}.png'.format(path,name,iters))

    x = (ground_true_draw.detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(x)
    img.save('{0}/rgb_gt.png'.format(path))