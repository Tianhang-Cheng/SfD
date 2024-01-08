'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import torch
import torch.nn as nn
import torch.nn.functional as torch_F
import tinycudann as tcnn
from functools import partial
import numpy as np

BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]], device='cuda')


class MLPforNeuralSDF(torch.nn.Module):

    def __init__(self, layer_dims, skip_connection=[], activ=None, use_layernorm=False, use_weightnorm=False,
                 geometric_init=False, out_bias=0., invert=False):
        """Initialize a multi-layer perceptron with skip connection.
        Args:
            layer_dims: A list of integers representing the number of channels in each layer.
            skip_connection: A list of integers representing the index of layers to add skip connection.
        """
        super().__init__()
        self.skip_connection = skip_connection
        self.use_layernorm = use_layernorm
        self.linears = torch.nn.ModuleList()
        if use_layernorm:
            self.layer_norm = torch.nn.ModuleList()
        # Hidden layers
        layer_dim_pairs = list(zip(layer_dims[:-1], layer_dims[1:]))
        for li, (k_in, k_out) in enumerate(layer_dim_pairs):
            if li in self.skip_connection:
                k_in += layer_dims[0]
            linear = torch.nn.Linear(k_in, k_out)
            if geometric_init:
                self._geometric_init(linear, k_in, k_out, first=(li == 0),
                                     skip_dim=(layer_dims[0] if li in self.skip_connection else 0))
            if use_weightnorm:
                linear = torch.nn.utils.weight_norm(linear)
            self.linears.append(linear)
            if use_layernorm and li != len(layer_dim_pairs) - 1:
                self.layer_norm.append(torch.nn.LayerNorm(k_out))
            if li == len(layer_dim_pairs) - 1:
                self.linears[-1].bias.data.fill_(0.0)
        # SDF prediction layer
        self.linear_sdf = torch.nn.Linear(k_in, 1)
        if geometric_init:
            self._geometric_init_sdf(self.linear_sdf, k_in, out_bias=out_bias, invert=invert)
        self.activ = activ or torch_F.relu_

    def forward(self, input, with_sdf=True, with_feat=True):
        feat = input
        for li, linear in enumerate(self.linears):
            if li in self.skip_connection:
                feat = torch.cat([feat, input], dim=-1)
            if li != len(self.linears) - 1 or with_feat:
                feat_pre = linear(feat)
                if self.use_layernorm:
                    feat_pre = self.layer_norm[li](feat_pre)
                feat_activ = self.activ(feat_pre)
            if li == len(self.linears) - 1:
                out = [self.linear_sdf(feat) if with_sdf else None,
                       feat_activ if with_feat else None]
            feat = feat_activ
        return out

    def _geometric_init(self, linear, k_in, k_out, first=False, skip_dim=0):
        torch.nn.init.constant_(linear.bias, 0.0)
        torch.nn.init.normal_(linear.weight, 0.0, np.sqrt(2 / k_out))
        if first:
            torch.nn.init.constant_(linear.weight[:, 3:], 0.0)  # positional encodings
        if skip_dim:
            torch.nn.init.constant_(linear.weight[:, -skip_dim:], 0.0)  # skip connections

    def _geometric_init_sdf(self, linear, k_in, out_bias=0., invert=False):
        torch.nn.init.normal_(linear.weight, mean=np.sqrt(np.pi / k_in), std=0.0001)
        torch.nn.init.constant_(linear.bias, -out_bias)
        if invert:
            linear.weight.data *= -1
            linear.bias.data *= -1

def get_activation(activ, **kwargs):
    func = dict(
        identity=lambda x: x,
        relu=torch_F.relu,
        relu_=torch_F.relu_,
        abs=torch.abs,
        abs_=torch.abs_,
        sigmoid=torch.sigmoid,
        sigmoid_=torch.sigmoid_,
        exp=torch.exp,
        exp_=torch.exp_,
        softplus=torch_F.softplus,
        silu=torch_F.silu,
        silu_=partial(torch_F.silu, inplace=True),
    )[activ]
    return partial(func, **kwargs)

def positional_encoding(input, num_freq_bases):
    """Encode input into position codes.
    Args:
        input (tensor [bs, ..., N]): A batch of data with N dimension.
        num_freq_bases: (int): The number of frequency base of the code.
    Returns:
        input_enc (tensor [bs, ..., 2*N*num_freq_bases]): Positional codes for input.
    """
    freq = 2 ** torch.arange(num_freq_bases, dtype=torch.float32, device=input.device) * np.pi  # [L].
    spectrum = input[..., None] * freq  # [B,...,N,L].
    sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L].
    input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L].
    input_enc = input_enc.view(*input.shape[:-1], -1)  # [B,...,2NL].
    return input_enc

class HashNeuralSDF(torch.nn.Module):

    def __init__(self, cfg_sdf):
        super().__init__()
        self.cfg_sdf = cfg_sdf
        encoding_dim = self.build_encoding(cfg_sdf.encoding)
        input_dim = 3 + encoding_dim
        self.build_mlp(cfg_sdf.mlp, input_dim=input_dim)

    def build_encoding(self, cfg_encoding):
        if cfg_encoding.type == "fourier":
            encoding_dim = 6 * cfg_encoding.levels
        elif cfg_encoding.type == "hashgrid":
            # Build the multi-resolution hash grid.
            l_min, l_max = cfg_encoding.hashgrid.min_logres, cfg_encoding.hashgrid.max_logres
            r_min, r_max = 2 ** l_min, 2 ** l_max
            num_levels = cfg_encoding.levels
            self.growth_rate = np.exp((np.log(r_max) - np.log(r_min)) / (num_levels - 1))
            config = dict(
                otype="HashGrid",
                n_levels=cfg_encoding.levels,
                n_features_per_level=cfg_encoding.hashgrid.dim,
                log2_hashmap_size=cfg_encoding.hashgrid.dict_size,
                base_resolution=2 ** cfg_encoding.hashgrid.min_logres,
                per_level_scale=self.growth_rate,
            )
            self.tcnn_encoding = tcnn.Encoding(3, config)
            self.resolutions = []
            for lv in range(0, num_levels):
                size = np.floor(r_min * self.growth_rate ** lv).astype(int) + 1
                self.resolutions.append(size)
            encoding_dim = cfg_encoding.hashgrid.dim * cfg_encoding.levels
        else:
            raise NotImplementedError("Unknown encoding type")
        return encoding_dim

    def build_mlp(self, cfg_mlp, input_dim=3):
        # SDF + point-wise feature
        layer_dims = [input_dim] + [cfg_mlp.hidden_dim] * cfg_mlp.num_layers + [cfg_mlp.hidden_dim]
        activ = get_activation(cfg_mlp.activ, **cfg_mlp.activ_params)
        self.mlp = MLPforNeuralSDF(layer_dims, skip_connection=cfg_mlp.skip, activ=activ,
                                   use_weightnorm=cfg_mlp.weight_norm, geometric_init=cfg_mlp.geometric_init,
                                   out_bias=cfg_mlp.out_bias, invert=cfg_mlp.inside_out)

    def forward(self, points_3D, transform_func=None, with_sdf=True, with_feat=True, input_variant_vector=None, trans_use_grad=True, direct=True):

        if transform_func is not None:
            transform_output = transform_func(points_3D, use_grad=trans_use_grad, direct=direct)
            input_local = transform_output['x_local']
            input_variant_vector = transform_output['vec_q']
        else: 
            input_local = points_3D

        # outside = torch.max(torch.abs(input_local), dim=-1)[0] > 1.2 # relaxed boundary
        points_enc = self.encode(input_local)  # [...,3+LD] , input_local range is [-1,1]
        sdf, feat = self.mlp(points_enc, with_sdf=with_sdf, with_feat=with_feat)
        # sdf[outside] += 1000 # use a large number to represent outside
        if with_feat:
            return torch.cat([sdf, feat], dim=-1)  # [...,1+K]
        else:
            return sdf  # [...,1] 

    def sdf(self, points_3D, transform_func=None, trans_use_grad=False):
        return self.forward(points_3D, transform_func=transform_func, trans_use_grad=trans_use_grad, with_sdf=True, with_feat=False)

    def encode(self, points_3D):
        if self.cfg_sdf.encoding.type == "fourier":
            points_enc = positional_encoding(points_3D, num_freq_bases=self.cfg_sdf.encoding.levels)
            feat_dim = 6
        elif self.cfg_sdf.encoding.type == "hashgrid":
            # Tri-linear interpolate the corresponding embeddings from the dictionary.
            vol_min, vol_max = self.cfg_sdf.encoding.hashgrid.range
            points_3D_normalized = (points_3D - vol_min) / (vol_max - vol_min)  # Normalize to [0,1].
            tcnn_input = points_3D_normalized.view(-1, 3)
            tcnn_output = self.tcnn_encoding(tcnn_input)
            points_enc = tcnn_output.view(*points_3D_normalized.shape[:-1], tcnn_output.shape[-1])
            feat_dim = self.cfg_sdf.encoding.hashgrid.dim
        else:
            raise NotImplementedError("Unknown encoding type")
        # Coarse-to-fine.
        if self.cfg_sdf.encoding.coarse2fine.enabled:
            mask = self._get_coarse2fine_mask(points_enc, feat_dim=feat_dim)
            points_enc = points_enc * mask
        points_enc = torch.cat([points_3D, points_enc], dim=-1)  # [B,R,N,3+LD]
        return points_enc

    def set_active_levels(self, current_iter, warm_up_end):
        anneal_levels = max((current_iter - warm_up_end) // self.cfg_sdf.encoding.coarse2fine.step, 1)
        self.anneal_levels = min(self.cfg_sdf.encoding.levels, anneal_levels)
        self.active_levels = max(self.cfg_sdf.encoding.coarse2fine.init_active_level, self.anneal_levels)

    def set_normal_epsilon(self):
        if self.cfg_sdf.encoding.coarse2fine.enabled:
            epsilon_res = self.resolutions[self.anneal_levels - 1]
        else:
            epsilon_res = self.resolutions[-1]
        self.normal_eps = 1. / epsilon_res

    @torch.no_grad()
    def _get_coarse2fine_mask(self, points_enc, feat_dim):
        mask = torch.zeros_like(points_enc)
        mask[..., :(self.active_levels * feat_dim)] = 1
        return mask
 
    def gradient(self, x, transform_func=None, is_training=False, is_training_geometry=False, **kwargs):

        result_dict = {'gradients_world': None, 'hessian': None}
        
        compute_hessian = is_training_geometry and is_training
        
        # Note: hessian is not fully hessian but diagonal elements
        if self.cfg_sdf.gradient.mode == "analytical":
            requires_grad = x.requires_grad
            with torch.enable_grad():
                # 1st-order gradient
                x.requires_grad_(True)
                sdf = self.sdf(x, transform_func=transform_func, trans_use_grad=False)
                gradient = torch.autograd.grad(sdf.sum(), x, create_graph=True, retain_graph=compute_hessian, only_inputs=True)[0]
                # 2nd-order gradient (hessian)
                if compute_hessian:
                    hessian = torch.autograd.grad(gradient.sum(), x, create_graph=True)[0]
                else:
                    hessian = None
                    gradient = gradient.detach()
                result_dict['gradients_world'] = gradient
                result_dict['hessian'] = hessian
            x.requires_grad_(requires_grad)
        elif self.cfg_sdf.gradient.mode == "numerical": 
            if self.cfg_sdf.gradient.taps == 6:
                eps = self.normal_eps
                # 1st-order gradient
                eps_x = torch.tensor([eps, 0., 0.], dtype=x.dtype, device=x.device)  # [3]
                eps_y = torch.tensor([0., eps, 0.], dtype=x.dtype, device=x.device)  # [3]
                eps_z = torch.tensor([0., 0., eps], dtype=x.dtype, device=x.device)  # [3]
                sdf_x_pos = self.sdf(x + eps_x)  # [...,1]
                sdf_x_neg = self.sdf(x - eps_x)  # [...,1]
                sdf_y_pos = self.sdf(x + eps_y)  # [...,1]
                sdf_y_neg = self.sdf(x - eps_y)  # [...,1]
                sdf_z_pos = self.sdf(x + eps_z)  # [...,1]
                sdf_z_neg = self.sdf(x - eps_z)  # [...,1]
                gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps)
                gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps)
                gradient_z = (sdf_z_pos - sdf_z_neg) / (2 * eps)
                gradient = torch.cat([gradient_x, gradient_y, gradient_z], dim=-1)  # [...,3]
                # 2nd-order gradient (hessian)
                if is_training:
                    sdf = kwargs.get('sdf', None)
                    assert sdf is not None  # computed when feed-forwarding through the network
                    hessian_xx = (sdf_x_pos + sdf_x_neg - 2 * sdf) / (eps ** 2)  # [...,1]
                    hessian_yy = (sdf_y_pos + sdf_y_neg - 2 * sdf) / (eps ** 2)  # [...,1]
                    hessian_zz = (sdf_z_pos + sdf_z_neg - 2 * sdf) / (eps ** 2)  # [...,1]
                    hessian = torch.cat([hessian_xx, hessian_yy, hessian_zz], dim=-1)  # [...,3]
                else:
                    hessian = None
            elif self.cfg_sdf.gradient.taps == 4:
                eps = self.normal_eps / np.sqrt(3)
                k1 = torch.tensor([1, -1, -1], dtype=x.dtype, device=x.device)  # [3]
                k2 = torch.tensor([-1, -1, 1], dtype=x.dtype, device=x.device)  # [3]
                k3 = torch.tensor([-1, 1, -1], dtype=x.dtype, device=x.device)  # [3]
                k4 = torch.tensor([1, 1, 1], dtype=x.dtype, device=x.device)  # [3]
                sdf1 = self.sdf(x + k1 * eps)  # [...,1]
                sdf2 = self.sdf(x + k2 * eps)  # [...,1]
                sdf3 = self.sdf(x + k3 * eps)  # [...,1]
                sdf4 = self.sdf(x + k4 * eps)  # [...,1]
                gradient = (k1*sdf1 + k2*sdf2 + k3*sdf3 + k4*sdf4) / (4.0 * eps)
                if is_training:
                    sdf = kwargs.get('sdf', None)
                    assert sdf is not None  # computed when feed-forwarding through the network
                    # the result of 4 taps is directly trace, but we assume they are individual components
                    # so we use the same signature as 6 taps
                    hessian_xx = ((sdf1 + sdf2 + sdf3 + sdf4) / 2.0 - 2 * sdf) / eps ** 2   # [N,1]
                    hessian = torch.cat([hessian_xx, hessian_xx, hessian_xx], dim=-1) / 3.0
                else:
                    hessian = None
            else:
                raise ValueError("Only support 4 or 6 taps.")
            result_dict['gradients_world'] = gradient
            result_dict['hessian'] = hessian
        return result_dict