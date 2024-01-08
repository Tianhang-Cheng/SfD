import torch
import torch.nn as nn
from torch.autograd import grad

import numpy as np

from model.embedder import get_embedder, mask_by_progress  

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size, # shape feature, pass gradient from shape to texture
            variant_vector_size, # model the variant of instances
            d_in, 
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            same_obj_num=1,
            use_numerical_gradient=False,
            use_hessian=False,
    ):
        super().__init__()
            
        d_out = 1
        dims = [d_in] + dims + [d_out + feature_vector_size]
 
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch 

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.progress = 1.0
        self.multires = multires
        self.same_obj_num = same_obj_num
        self.use_numerical_gradient = use_numerical_gradient
        self.use_analytical_gradient = not use_numerical_gradient
        self.use_hessian = use_hessian
        
        for l in range(0, self.num_layers - 1): 
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0] - variant_vector_size
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
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] + variant_vector_size - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)

    def forward(self, coordinates, transform_func=None, input_variant_vector=None, trans_use_grad=True, direct=True): 
        """
        input_variant_vector: (optional) if transform_func is not None, this will be ignored
        """

        if transform_func is not None:
            transform_output = transform_func(coordinates, use_grad=trans_use_grad, direct=direct)
            input_local = transform_output['x_local']
            input_variant_vector = transform_output['vec_q']
        else: 
            input_local = coordinates

        if self.embed_fn is not None:
            input_local = self.embed_fn(input_local) 
        
        input_local = mask_by_progress(input_local, multires=self.multires, progress=self.progress)

        x = input_local
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l)) 
            if l in self.skip_in: 
                if input_variant_vector is None:
                    x = torch.cat([x, input_local], 1) / np.sqrt(2) 
                else:
                    x = torch.cat([x, input_local, input_variant_vector], 1) / np.sqrt(2) 
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)
        return x

    def sdf(self, input_world, transform_func=None, input_variant_vector=None, direct=True, trans_use_grad=True): 
        return self.forward(input_world, transform_func, input_variant_vector=input_variant_vector, direct=direct, trans_use_grad=trans_use_grad)[..., 0] 

    def gradient(self, x_world, transform_func=None, input_variant_vector=None, is_training_geometry=False, is_training=False, **kwargs):
        """
        return world gradient
        save_memory: if True, we will not compute hessian
        """ 

        if input_variant_vector is not None:
            input_variant_vector = input_variant_vector.detach()

        result_dict = {'gradients_world': None, 'hessian': None}

        # our original pipeline
        if self.use_analytical_gradient:
            x_world.requires_grad_(True)  
            # y = self.forward(x_world, transform_func)[:,:1]  # sdf value
            # if transform_func is not None:
            #     x_local = transform_func(x_world, use_grad=True)[0]
            # else:
            #     x_local = x_world
            
            y = self.forward(x_world, transform_func, input_variant_vector=input_variant_vector, trans_use_grad=False)[:, :1]  # sdf value 
            gradients_world = torch.autograd.grad(y.sum(), x_world, create_graph=True, only_inputs=True)[0]
            result_dict['gradients_world'] = gradients_world

            if is_training and is_training_geometry and self.use_hessian:
                hessian_world = torch.autograd.grad(outputs=gradients_world.sum(), inputs=x_world, create_graph=True)[0]
                result_dict['hessian'] = hessian_world

        # ablation study: use numerical gradient
        # adapted from https://github.com/NVlabs/neuralangelo/blob/main/projects/neuralangelo/model.py
        elif self.use_numerical_gradient:
            raise NotImplementedError
            # if eval, we don't need to enable grad
            with torch.set_grad_enabled(is_training):
                eps = 1e-1
                x = x_world
                k1 = torch.tensor([1, -1, -1], dtype=x.dtype, device=x.device)  # [3]
                k2 = torch.tensor([-1, -1, 1], dtype=x.dtype, device=x.device)  # [3]
                k3 = torch.tensor([-1, 1, -1], dtype=x.dtype, device=x.device)  # [3]
                k4 = torch.tensor([1, 1, 1], dtype=x.dtype, device=x.device)  # [3]
                
                sdf1 = self.forward(x + k1 * eps, transform_func, input_variant_vector=input_variant_vector, trans_use_grad=False)[:, :1]  # sdf value
                sdf2 = self.forward(x + k2 * eps, transform_func, input_variant_vector=input_variant_vector, trans_use_grad=False)[:, :1]  # [...,1]
                sdf3 = self.forward(x + k3 * eps, transform_func, input_variant_vector=input_variant_vector, trans_use_grad=False)[:, :1]  # [...,1]
                sdf4 = self.forward(x + k4 * eps, transform_func, input_variant_vector=input_variant_vector, trans_use_grad=False)[:, :1]  # [...,1]

                # x_local = transform_func(x + k1 * eps, use_grad=False)['x_local']
                # sdf1 = self.forward(x_local + k1 * eps, input_variant_vector=input_variant_vector)[:, :1]  # sdf value
                # sdf2 = self.forward(x_local + k2 * eps, input_variant_vector=input_variant_vector)[:, :1]  # [...,1]
                # sdf3 = self.forward(x_local + k3 * eps, input_variant_vector=input_variant_vector)[:, :1]  # [...,1]
                # sdf4 = self.forward(x_local + k4 * eps, input_variant_vector=input_variant_vector)[:, :1]  # [...,1]

                gradients_world = (k1*sdf1 + k2*sdf2 + k3*sdf3 + k4*sdf4) / (4.0 * eps)

                # assert sdf is not None  # computed when feed-forwarding through the network
                # the result of 4 taps is directly trace, but we assume they are individual components
                # so we use the same signature as 6 taps
                with torch.no_grad():
                    sdf = self.forward(x, transform_func, input_variant_vector=input_variant_vector, trans_use_grad=False)[:, :1]
                    # sdf = self.forward(x_local, input_variant_vector=input_variant_vector)[:, :1]
                hessian_xx = ((sdf1 + sdf2 + sdf3 + sdf4) / 2.0 - 2 * sdf) / eps ** 2   # [N,1]
                hessian = torch.cat([hessian_xx, hessian_xx, hessian_xx], dim=-1) / 3.0

            result_dict['hessian'] = hessian
            result_dict['gradients_world'] = gradients_world

        return result_dict