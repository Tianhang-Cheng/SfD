import sys
sys.path.append('../Dup') 

import torch
from torch import nn
from torch.nn import functional as F
from model.embedder import get_embedder
import numpy as np

pi = np.pi
CE = nn.CrossEntropyLoss()
clip_img = lambda x: torch.clamp(x, min=0., max=1.)

def get_seg_loss2(seg_values, seg_gt): 
    target_sdf = seg_values[torch.arange(seg_values.shape[0]), seg_gt] # target should be the lowest sdf
    wrong_sdf = (seg_values - target_sdf.unsqueeze(-1)) # should > 0
    seg_loss = torch.relu(-wrong_sdf).mean() # penalize < 0
    # seg_loss = CE(-seg_values, seg_gt) / float(seg_values.shape[0])

    return seg_loss

class PointTransformationLoss(nn.Module):
    def __init__(self, sdf_supervision_weight, instance_seg_weight):
        super().__init__()

        self.sdf_supervision_weight = sdf_supervision_weight 
        self.instance_seg_weight = instance_seg_weight  
    
    def forward(self, model_outputs):

        """
        pred_prob [batch, c]
        sdf gt_class [batch]
        data gt_class [batch]
        """  

        # mask
        # network_object_mask = model_outputs['network_object_mask'].bool()
        # object_mask = model_outputs['object_mask'].bool()
        
        x = model_outputs['select_proxy_sdf']
        y = model_outputs['select_network_sdf']
        z = model_outputs['select_gt_seg']

        # proxy sdf - network sdf loss 
        # distillation_loss = torch.tensor(0.0).cuda()
        distillation_loss = F.mse_loss(x, y.detach()) * self.sdf_supervision_weight
 
        # network segmentation - real segmentation loss 
        # segmentation_loss = torch.tensor(0.0).cuda()
        segmentation_loss = (get_seg_loss2(y, z.reshape(-1)) + get_seg_loss2(x, z.reshape(-1))) * self.instance_seg_weight   

        # total loss 
        total_loss = distillation_loss + segmentation_loss  

        ret = {'transformation_total_loss': total_loss,
               'distillation_loss':distillation_loss,
               'segmentation_loss': segmentation_loss,
               }

        return ret

class NeuSLoss(nn.Module):
    def __init__(self, mask_weight, eikonal_weight, normal_weight, loss_type, use_pretrain_normal=False):
        super().__init__()
 
        self.mask_weight = mask_weight
        self.normal_weight = normal_weight
        self.eikonal_weight = eikonal_weight
        self.hessian_weight = 5e-4 
        self.use_pretrain_normal = use_pretrain_normal
        self.colmap_sdf_weight = 1

        if loss_type == 'L1':
            print('Using L1 loss for comparing images!')
            self.img_loss = nn.L1Loss(reduction='mean')
        elif loss_type == 'L2':
            print('Using L2 loss for comparing images!')
            self.img_loss = nn.MSELoss(reduction='mean')
        else:
            raise Exception('Unknown loss_type!') 

    def get_normal_loss(self, normal_pred, normal_gt):
        normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
        normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
        l1 = F.smooth_l1_loss(input=normal_pred, target=normal_gt) * 3
        cos = (1. - torch.sum(normal_pred * normal_gt, dim = -1)).mean()
        return (l1 + cos) * 0.05

    def forward(self, model_output, ground_truth, **kwargs):  
        object_mask = ground_truth['mask'] 
        network_mask = model_output['network_object_mask'] 
        mask = (object_mask[:, 0] > 0).bool() 
        weights_sum = model_output['weights_sum'] 
        weights_sum = weights_sum.clip(1e-3, 1.0 - 1e-4)
        normal = model_output['surface_normal'] 
        pred_color = model_output['color'] 
        gt_color = ground_truth['color'] 
 
        color_error1 = self.img_loss(pred_color[network_mask], gt_color[network_mask]) # use L1 for more robust result, L2 for better result is pose is good
        color_error2 = self.img_loss(pred_color[~mask], torch.zeros_like(pred_color[~mask]).cuda()) * 0.5
        color_error = color_error1 + color_error2
        eikonal_loss = model_output['gradient_error'] * self.eikonal_weight
        hessian_loss = model_output['hessian_error'] * self.hessian_weight
        colmap_sdf_loss = model_output['colmap_sdf_error'] * self.colmap_sdf_weight
        mask_loss = F.binary_cross_entropy(weights_sum, object_mask) * self.mask_weight 

        normal_loss = torch.tensor(0.0).cuda() 
        if self.use_pretrain_normal: 
            normal_camera = torch.matmul(torch.linalg.inv(ground_truth['pose'][:, 0:3, 0:3]), normal[:, :, None])[..., 0]
            # normal_camera = normal_camera * torch.tensor([1, -1, -1]).cuda()
            normal_camera = clip_img((normal_camera + 1) / 2.0)
            gt_normal = ground_truth['pretrain_normal']
            gt_normal = gt_normal / torch.norm(gt_normal, dim=-1, keepdim=True) # FIXME: need to normalize?
            gt_normal = gt_normal[mask & network_mask]
            normal_camera = normal_camera[mask & network_mask]
            normal_loss = self.get_normal_loss(normal_camera, gt_normal) * self.normal_weight

        loss = color_error + eikonal_loss + mask_loss + normal_loss + hessian_loss + colmap_sdf_loss

        ret = {'neus_total_loss': loss,
               'color_loss': color_error, 
               'eikonal_loss': eikonal_loss,
               'hessian_loss': hessian_loss,
               'colmap_sdf_loss': colmap_sdf_loss,
               'mask_loss': mask_loss,
               'normal_loss': normal_loss,  
               }
        
        return ret
    
class NeuSVisLoss(nn.Module):
    def __init__(self):
        super().__init__()
  
    def forward(self, model_output): 

        pred_vis = model_output['pred_vis']
        gt_vis = model_output['gt_vis']
        network_object_mask = model_output['network_object_mask']

        # compute visibility loss 
        if network_object_mask.sum() > 0:
            pred_vis = pred_vis[network_object_mask].float().reshape(-1, 2)
            gt_vis = (~gt_vis[network_object_mask]).long().reshape(-1)
            visibility_loss = nn.CrossEntropyLoss()(pred_vis, gt_vis) 
        else:
            visibility_loss = torch.tensor([0.0]).cuda()
 
        ret = {'visibility_loss': visibility_loss }

        return ret

class NeuSMatLoss(nn.Module):
    def __init__(self, brdf_multires, sg_rgb_weight, kl_weight, latent_smooth_weight, metal_reg_weight=0.01, loss_type='L1'):
        super().__init__()

        self.brdf_multires = brdf_multires
        self.sg_rgb_weight = sg_rgb_weight
        self.kl_weight = kl_weight
        self.latent_smooth_weight = latent_smooth_weight
        self.loss_type = loss_type
        self.metal_reg_weight = metal_reg_weight

        if loss_type == 'L1':
            print('Using L1 loss for comparing images!')
            self.img_loss = nn.L1Loss(reduction='mean')
        elif loss_type == 'L2':
            print('Using L2 loss for comparing images!')
            self.img_loss = nn.MSELoss(reduction='mean') 
        else:
            raise Exception('Unknown loss_type!')
    
    def get_rgb_loss(self,rgb_values, rgb_gt, network_object_mask, object_mask):

        union_mask = object_mask & network_object_mask
        # union_mask = object_mask
        if (union_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[union_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[union_mask] 
        
        rgb_loss = self.img_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_latent_smooth_loss(self, model_outputs):
        d_diff = model_outputs['diffuse_albedo']
        d_rough = model_outputs['roughness'][..., 0]
        d_xi_diff = model_outputs['random_xi_diffuse_albedo']
        d_xi_rough = model_outputs['random_xi_roughness'][..., 0]
        loss = nn.L1Loss()(d_diff, d_xi_diff) + nn.L1Loss()(d_rough, d_xi_rough) 
        return loss 

    def kl_divergence(self, rho, rho_hat):
        rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
        rho = torch.tensor([rho] * len(rho_hat)).cuda()
        return torch.mean(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))

    def get_kl_loss(self, model, points ):
        loss = 0
        if self.brdf_multires > 0:
            embed_fn, _ = get_embedder(self.brdf_multires)
            values = embed_fn(points)
        else:
            values = points

        for i in range(len(model.brdf_encoder_layer)):
            values = model.brdf_encoder_layer[i](values) 
        
        if len(values) == 0:
            loss += torch.tensor([0.0]).cuda()
        else:
            loss += self.kl_divergence(0.05, values)

        return loss
    
    def get_metal_loss(self, metallic):
        loss = metallic * (1.0 - metallic) 
        return loss.mean() 
    
    def forward(self, model_output, object_mask, rgb_gt, mat_model): 
        network_object_mask = model_output['network_object_mask'] 
        pred_rgb = model_output['sg_rgb']
        sg_rgb_loss = self.get_rgb_loss(pred_rgb, rgb_gt, network_object_mask, object_mask)

        latent_smooth_loss = self.get_latent_smooth_loss(model_output) 
        kl_loss = self.get_kl_loss(mat_model, model_output['points_local'][network_object_mask])
        metal_loss = self.get_metal_loss(model_output['metallic']) * self.metal_reg_weight 

        sg_total_loss = \
            self.sg_rgb_weight * sg_rgb_loss + \
            self.kl_weight * kl_loss + \
            self.latent_smooth_weight * latent_smooth_loss + metal_loss 
 
        output = {
            'sg_rgb_loss': sg_rgb_loss,
            'kl_loss': kl_loss,
            'latent_smooth_loss': latent_smooth_loss,  
            'sg_total_loss': sg_total_loss,
            'metal_loss': metal_loss,
            }
        return output