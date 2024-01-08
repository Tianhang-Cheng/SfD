import torch
import torch.nn as nn

from model.embedder import get_embedder, mask_by_progress

class MappingNetwork(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, use_leaky_relu=True):
        super().__init__()

        if use_leaky_relu:
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.ReLU()
        self.network = nn.Sequential(nn.Linear(d_in, d_hidden),
                                     self.act,
                                     nn.Linear(d_hidden, d_hidden),
                                     self.act,
                                     nn.Linear(d_hidden, d_hidden),
                                     self.act,
                                     nn.Linear(d_hidden, d_out))

    def forward(self, z):
        style = self.network(z) 
        return style
    
class FourierColorNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            multires_point=4,  
    ):
        super().__init__()

        self.progress = 1.0

        self.embedview_fn = lambda x: x
        self.multires_view = multires_view
        if multires_view > 0:
            embedview_fn, input_ch_view = get_embedder(multires_view)
            self.embedview_fn = embedview_fn 
        
        self.embedpoint_fn = lambda x: x
        self.multires_point = multires_point
        if multires_point > 0:
            embedpoint_fn, input_ch_point = get_embedder(multires_point)
            self.embedpoint_fn = embedpoint_fn
        
        self.embednormal_fn = lambda x: x
        self.multires_normal = multires_view
        if multires_view > 0:
            embednormal_fn, input_ch_normal = get_embedder(multires_view)
            self.embednormal_fn = embednormal_fn
        
        self.softplus = nn.Softplus()
        
        render_dims = [6 + feature_vector_size] + dims + [d_out]
 
        d_style = 1
        self.mapping_network = MappingNetwork(d_in=feature_vector_size + input_ch_point + input_ch_view + input_ch_view, d_hidden=128, d_out=d_style)
        self.render_num_layers = len(render_dims) - 1
        for l in range(0, self.render_num_layers):
            out_dim = render_dims[l + 1]
            lin = nn.Linear(render_dims[l], out_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
           
        self.relu = nn.ReLU()

        if feature_vector_size > 0: 
            self.embed_fn, input_dim = get_embedder(6) 
            linears = []
            dim = input_dim
            for _ in range(3):
                linears.append(nn.Linear(dim, 128))
                linears.append(self.relu)
                dim = 128
            linears.append(nn.Linear(dim, 3))
            self.linears = nn.Sequential(*linears)
        else:
            self.linears = None
 
    def forward(self, points, normals, view_dirs, feature_vectors, points_world=None, normals_world=None, view_dirs_world=None):
         
        if feature_vectors is None:
            feature_vectors = torch.zeros([points.shape[0], 0]).cuda()

        points_world = self.embedpoint_fn(points_world)
        view_dirs_world = self.embedview_fn(view_dirs_world)
        normals_world = self.embedview_fn(normals_world)

        points_world = mask_by_progress(points_world, multires=self.multires_point, progress=self.progress) 
        normals_world = mask_by_progress(normals_world, multires=self.multires_normal, progress=self.progress) 
        view_dirs_world = mask_by_progress(view_dirs_world, multires=self.multires_normal, progress=self.progress) 
        points = mask_by_progress(points, multires=self.multires_point, progress=self.progress) 
        normals = mask_by_progress(normals, multires=self.multires_normal, progress=self.progress)

        style_input = torch.cat([points_world, view_dirs_world, normals_world, feature_vectors], dim=-1)
        rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
 
        shadow = self.mapping_network(style_input)[..., 0:1] 
        shadow = torch.relu(shadow+1.0) + 1e-2
        x = rendering_input
        for l in range(0, self.render_num_layers):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.render_num_layers - 1:
                x = self.relu(x)
        albedo = torch.sigmoid(x[..., 0:3]) 
        x = albedo * shadow 
        
        res = {'color': x, 'albedo': albedo}
        return res