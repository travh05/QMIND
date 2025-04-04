import numpy as np
import torch
import torch.nn as nn

from deform_diffeo_lib.ops import InterpolateRescaleFieldPointwise
# from deform_diffeo_lib.ops.airlab_ops import BsplineTransformation
from deform_diffeo_lib.ops.airlab_replacement_ops import BsplineConv3dSeparable

class DenseVelocityField(nn.Module):
    def __init__(self, img_shape, sigma=[3,3,3], order=3, n_batch=1, dtype=torch.get_default_dtype(), device='cuda', allow_autodiff=False):
        '''
        img_shape: (h, w, d)
        '''
        super(DenseVelocityField, self).__init__()

        self.img_shape = img_shape

        if np.allclose(np.array(sigma), np.array([1,1,1])):
            self.cp_grid = torch.nn.Parameter(torch.empty([1,3,*img_shape], dtype=dtype, device=device).uniform_(-1e-3, 1e-3))
            self.transformation = lambda x: x # just return self.cp_grid
        else:
            self.transformation = BsplineConv3dSeparable(
                img_shape,
                sigma=sigma,
                order=order,
                dtype=dtype,
                device=device,
            )
            self.cp_grid = torch.nn.Parameter(torch.empty(self.transformation.cp_grid_shape, dtype=dtype, device=device).uniform_(-1e-3, 1e-3))

        self.interpolate_fxn = InterpolateRescaleFieldPointwise(img_shape, dtype=dtype, device=device)
        self.allow_autodiff = allow_autodiff

    def forward(self):
        '''
        compute self.dense_velocity_field, and return
        '''
        self.dense_velocity_field = self.transformation(self.cp_grid)
        return self.dense_velocity_field

    def get_dense_velocity_field(self):
        '''
        convenience method to keep the same syntax across different models
        '''
        self.dense_velocity_field = self.transformation(self.cp_grid)
        return self.dense_velocity_field

    def update_dense_velocity_field(self):
        '''
        convenience method to keep the same syntax across different models
        '''
        self.dense_velocity_field = self.transformation(self.cp_grid)

    def get_velocity_at_verts(self, verts_eval):
        '''
        verts_eval: torch.Tensor (n_verts, 3)

        1. self.dense_velocity_field should be computed once using "forward/get_dense_velocity" before velocity integration
        2. "get_velocity_at_verts" is called multiple times during velocity integration

        assume dense_velocity_field uses grid_sample conventions
        need velocity rescaled to img_dim_size b/c verts_eval needs to be re-calculated in xyz coordinates
        - this is also consistent with implicit representation, which outputs velocity scaled to img_dim_size
        '''
        velocity_at_verts = self.interpolate_fxn(self.dense_velocity_field, verts_eval, allow_autodiff=self.allow_autodiff)
        
        return velocity_at_verts