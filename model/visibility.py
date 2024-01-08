import torch
import torch.nn as nn

import numpy as np 

from model.embedder import get_embedder


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

        self.register_parameter('freq', nn.Parameter(torch.ones([1]), requires_grad=True))
        self.register_parameter('phase_shift', nn.Parameter(torch.zeros([1]), requires_grad=True))

    def forward(self, x):
        """
        x: [b,3]
        """
        x = self.layer(x)
        freq = self.freq * 15.0 + 30.0
        phase = torch.clip(self.phase_shift, -np.pi, np.pi)
        return torch.sin(freq * x + phase)

def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init

class SirenVisNetwork(nn.Module):
    def __init__(self, dims=[256, 256, 256, 256]):
        super().__init__() 

        self.input_dim = 6
        self.output_dim = 2

        vis_layer = []
        dim = self.input_dim
        for i in range(len(dims)):
            vis_layer.append(FiLMLayer(dim, dims[i]))
            dim = dims[i] 
        vis_layer.append(nn.Linear(dim, self.output_dim))
        self.vis_layer = nn.Sequential(*vis_layer)
        self.vis_layer.apply(frequency_init(25))

    def forward(self, points_world, view_dirs_world):

        """
        vis [b, 2]
        """
        x = torch.cat([points_world, view_dirs_world], dim=-1)
        vis = self.vis_layer(x)
        return vis

class VisNetwork(nn.Module):
    def __init__(self, points_multires=10, dirs_multires=4, dims=[128, 128, 128, 128]):
        super().__init__()
        p_input_dim = 3
        self.p_embed_fn = None
        if points_multires > 0:
            self.p_embed_fn, p_input_dim = get_embedder(points_multires)
        
        dir_input_dim = 3
        self.dir_embed_fn = None
        if dirs_multires > 0:
            self.dir_embed_fn, dir_input_dim = get_embedder(dirs_multires)

        self.actv_fn = nn.ReLU()

        vis_layer = []
        dim = p_input_dim + dir_input_dim
        for i in range(len(dims)):
            vis_layer.append(nn.Linear(dim, dims[i]))
            vis_layer.append(self.actv_fn)
            dim = dims[i]
        vis_layer.append(nn.Linear(dim, 2))
        self.vis_layer = nn.Sequential(*vis_layer)

    def forward(self, points, view_dirs, inv_pose=None):
        if self.p_embed_fn is not None:
            points = self.p_embed_fn(points)
        if self.dir_embed_fn is not None:
            view_dirs = self.dir_embed_fn(view_dirs)

        vis = self.vis_layer(torch.cat([points, view_dirs], -1))

        return vis