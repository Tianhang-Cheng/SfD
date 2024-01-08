import torch
import torch.nn as nn
import numpy as np

from model.geometry_triplane_util import WindowAttention, window_partition, window_reverse
from model.embedder import get_embedder, mask_by_progress  
from timm.models.layers import to_2tuple

def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1)
    iy = ((iy + 1) / 2) * (IH-1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.contiguous().view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

def generate_planes():
    return torch.tensor([[[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]],
                         [[0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0]],
                         [[0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]]], dtype=torch.float32, device='cuda')

def project_onto_planes(planes, coordinates):
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    coordinates = (2/box_warp) * coordinates

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = grid_sample(plane_features, projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)

    return output_features

class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        grid_resolution,             # Output resolution.
        grid_channels,               # Number of output color channels.
        rendering_kwargs = {},
        use_hessian=False,
    ):
        super().__init__()
        self.grid_resolution = grid_resolution
        self.grid_channels = grid_channels
        self.rendering_kwargs = rendering_kwargs
        self.tritype = 0
        self.progress = 1.0

        self.use_hessian = use_hessian

        sdf_para = SDFNetwork(d_in=3, d_out=24, d_hidden=256, n_layers=5)
        self.decoder = OSG_PE_SDFNetwork(d_in=288, d_out=257, d_hidden=256, n_layers=3, multires=self.rendering_kwargs['PE_res'], geometric_init=True)

        self._last_planes = None

        self.plane_axes = generate_planes()

        ini_sdf = torch.randn([3, self.grid_channels, self.grid_resolution, self.grid_resolution])
        xs = (torch.arange(self.grid_resolution) - (self.grid_resolution / 2 - 0.5)) / (self.grid_resolution / 2 - 0.5)
        ys = (torch.arange(self.grid_resolution) - (self.grid_resolution / 2 - 0.5)) / (self.grid_resolution / 2 - 0.5)
        (ys, xs) = torch.meshgrid(-ys, xs)
        N = self.grid_resolution
        zs = torch.zeros(N, N)
        inputx = torch.stack([zs, xs, ys]).permute(1, 2, 0).reshape(N ** 2, 3)
        inputy = torch.stack([xs, zs, ys]).permute(1, 2, 0).reshape(N ** 2, 3)
        inputz = torch.stack([xs, ys, zs]).permute(1, 2, 0).reshape(N ** 2, 3)
        ini_sdf[0] = sdf_para(inputx).permute(1, 0).reshape(self.grid_channels, N, N)
        ini_sdf[1] = sdf_para(inputy).permute(1, 0).reshape(self.grid_channels, N, N)
        ini_sdf[2] = sdf_para(inputz).permute(1, 0).reshape(self.grid_channels, N, N)

        self.planes = torch.nn.Parameter(ini_sdf.unsqueeze(0), requires_grad=True)

        self.window_size = self.rendering_kwargs['attention_window_size']
        self.numheads = self.rendering_kwargs['attention_numheads']
        self.attn = WindowAttention(self.grid_channels, window_size=to_2tuple(self.window_size), num_heads=self.numheads)
        self.window_size4 = self.window_size * 2
        self.attn4 = WindowAttention(self.grid_channels, window_size=to_2tuple(self.window_size4), num_heads=self.numheads)
        self.window_size2 = self.window_size // 2
        self.attn2 = WindowAttention(self.grid_channels, window_size=to_2tuple(self.window_size2), num_heads=self.numheads)

        self.embed_fn = get_embedder(rendering_kwargs['multiply_PE_res'], input_dims=3, include_input=False)[0]

    def forward(self, coordinates, transform_func=None, input_variant_vector=None, trans_use_grad=True, direct=True):
        planes = self.planes
        planes = planes.view(len(planes), 3, planes.shape[-3], planes.shape[-2], planes.shape[-1])
        
        if transform_func is not None:
            transform_output = transform_func(coordinates, use_grad=trans_use_grad, direct=direct)
            input_local = transform_output['x_local']
            input_variant_vector = transform_output['vec_q']
        else: 
            input_local = coordinates
        return self.run_model(planes, self.decoder, input_local.unsqueeze(0), self.rendering_kwargs)

    def run_model(self, planes, decoder, sample_coordinates, options):
        grid_channels = self.grid_channels
        sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])
        planes_attention = planes.squeeze(0).view(3, planes.shape[-3], planes.shape[-2], planes.shape[-1]).permute(0, 2, 3, 1)
        x_windows = window_partition(planes_attention, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, grid_channels)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, grid_channels)
        shifted_x = window_reverse(attn_windows, self.window_size, planes.shape[-2], planes.shape[-1])
        planes_attention = shifted_x.permute(0, 3, 1, 2).unsqueeze(0)
        sampled_features_attention = sample_from_planes(self.plane_axes, planes_attention, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])
        planes_attention = planes.squeeze(0).view(3, planes.shape[-3], planes.shape[-2], planes.shape[-1]).permute(0, 2, 3, 1)
        x_windows = window_partition(planes_attention, self.window_size4)
        x_windows = x_windows.view(-1, self.window_size4 * self.window_size4, grid_channels)
        attn_windows = self.attn4(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size4, self.window_size4, grid_channels)
        shifted_x = window_reverse(attn_windows, self.window_size4, planes.shape[-2], planes.shape[-1])
        planes_attention = shifted_x.permute(0, 3, 1, 2).unsqueeze(0)
        sampled_features_attention4 = sample_from_planes(self.plane_axes, planes_attention, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])
        planes_attention = planes.squeeze(0).view(3, planes.shape[-3], planes.shape[-2], planes.shape[-1]).permute(0, 2, 3, 1)
        x_windows = window_partition(planes_attention, self.window_size2)
        x_windows = x_windows.view(-1, self.window_size2 * self.window_size2, grid_channels)
        attn_windows = self.attn2(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size2, self.window_size2, grid_channels)
        shifted_x = window_reverse(attn_windows, self.window_size2, planes.shape[-2], planes.shape[-1])
        planes_attention = shifted_x.permute(0, 3, 1, 2).unsqueeze(0)
        sampled_features_attention2 = sample_from_planes(self.plane_axes, planes_attention, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])

        sampled_features = torch.cat([sampled_features_attention4, sampled_features_attention, sampled_features_attention2, sampled_features], dim=-1)

        p_encode = self.embed_fn(sample_coordinates)
        d = sampled_features.shape[-1] // (p_encode.shape[-1] // 3)
        x = p_encode.view(1, -1, 4, options['multiply_PE_res']//4*2, 3)[:, :, :, :, 0]
        y = p_encode.view(1, -1, 4, options['multiply_PE_res']//4*2, 3)[:, :, :, :, 1]
        z = p_encode.view(1, -1, 4, options['multiply_PE_res']//4*2, 3)[:, :, :, :, 2]
        p_encode = torch.cat([z, x, y]).tile(1, 1, d).view(3, p_encode.shape[1], -1)
        sampled_features = sampled_features * p_encode.unsqueeze(0)
        _, dim, N, nf = sampled_features.shape
        out = decoder(sampled_features, sample_coordinates)
        return out

    def sdf(self, input_world, transform_func=None, input_variant_vector=None, trans_use_grad=True, direct=True): 
        return self.forward(input_world, transform_func, input_variant_vector=input_variant_vector, trans_use_grad=trans_use_grad, direct=direct)[..., :1] 

    def gradient(self, x_world, transform_func=None, input_variant_vector=None, save_memory=False, is_training=False, **kwargs):

        if input_variant_vector is not None:
            input_variant_vector = input_variant_vector.detach()

        result_dict = {'gradients_world': None, 'hessian': None}

        # our original pipeline 
        x_world.requires_grad_(True)  
        # y = self.forward(x_world, transform_func)[:,:1]  # sdf value
        # if transform_func is not None:
        #     x_local = transform_func(x_world, use_grad=True)[0]
        # else:
        #     x_local = x_world
        
        y = self.forward(x_world, transform_func, input_variant_vector=input_variant_vector, trans_use_grad=False)[:, :1]  # sdf value
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients_world = torch.autograd.grad(
            outputs=y,
            inputs=x_world,
            grad_outputs=d_output,
            create_graph=is_training and not save_memory,
            retain_graph=is_training and not save_memory,
            only_inputs=True
        )[0]
        result_dict['gradients_world'] = gradients_world

        if is_training and self.use_hessian:
            hessian_world = torch.autograd.grad(
                outputs=gradients_world.sum(),
                inputs=x_world,
                create_graph=True,
            )[0]
            result_dict['hessian'] = hessian_world

        return result_dict

class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 geometric_init=True,
                 weight_norm=True):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        self.multires = multires

        self.num_layers = len(dims)
        self.skip_in = skip_in 

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs): 

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 1:
                x = self.activation(x)
        return x


class OSG_PE_SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(10,),
                 multires=0,
                 bias=0.5,
                 geometric_init=True,
                 weight_norm=True, 
                 use_hessian=False,
                 use_numerical_gradient=False,):
        super(OSG_PE_SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out] 
  
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = d_in + input_ch
        else:
            dims[0] = d_in + 3

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.progress = 1.0
        self.multires = multires 
        self.use_numerical_gradient = use_numerical_gradient
        self.use_analytical_gradient = not use_numerical_gradient
        self.use_hessian = use_hessian
        
        for l in range(0, self.num_layers - 1): 
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]  
                assert out_dim > 0
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0]  - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)

    def forward(self, inputs_feat, inputs_PE, input_variant_vector=None):
        assert input_variant_vector is None
        _, dim, N, nf = inputs_feat.shape
        inputs_feat = inputs_feat.squeeze(0).permute(1, 2, 0).reshape(N, nf*dim)
        inputs_PE = self.embed_fn(inputs_PE)
        inputs_PE = mask_by_progress(inputs_PE, multires=self.multires, progress=self.progress).squeeze(0)

        inputs = torch.cat([inputs_PE, inputs_feat], dim=-1)
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l)) 
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)  
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x) 
        return x