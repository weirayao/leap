import torch
import torch.nn as nn
from torch.nn import functional as F, init
import torch.distributions as D
from .components.transforms import (AfflineCoupling,
                                    AffineMBD,
                                    ComponentWiseSpline)

class TemporalMBDFlow(nn.Module):
    def __init__(self, input_size=2, lags= 4):
        super().__init__()
        self.L = lags
        self.D = input_size
        self.unmix = AfflineCoupling(n_blocks = 3, 
                                     input_size = input_size, 
                                     hidden_size = 64, 
                                     n_hidden = 1, 
                                     batch_norm=True)
        self.dconv = AffineMBD(input_size = input_size, 
                               lags = lags)
        self.spline = ComponentWiseSpline(input_dim = input_size)
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.eye(input_size))

    @property
    def base_dist(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y):
        # x: [BS, T, D], y: [BS, L, D]
        x_shape, y_shape = x.shape, y.shape
        xx = x.reshape(-1, self.D)
        yy = y.reshape(-1, self.D)
        sum_log_abs_det_jacobians = 0
        ly, _ = self.unmix(yy)
        ly = ly.reshape(y_shape)
        lx, logabsdet = self.unmix(xx)
        lx = lx.reshape(x_shape)
        logabsdet = torch.sum(logabsdet.reshape(x_shape[0],x_shape[1]), dim=1)
        sum_log_abs_det_jacobians += logabsdet
        # xx: [BS, T+L, D], e: [BS, T, D]
        xx = torch.cat((ly, lx), dim=1)
        e, logabsdet  = self.dconv(xx)
        sum_log_abs_det_jacobians += logabsdet
        ee = e.reshape(-1, self.D)
        z, logabsdet = self.spline(ee)
        z = z.reshape(x_shape)
        logabsdet = torch.sum(logabsdet.reshape(x_shape[0],x_shape[1]), dim=1)
        sum_log_abs_det_jacobians += logabsdet
        return e, z, sum_log_abs_det_jacobians

    def inverse(self, z, y):
        # z: [BS, T, D], y: [BS, L, D]
        z_shape, y_shape = z.shape, y.shape
        sum_log_abs_det_jacobians = 0
        zz = z.reshape(-1, self.D)
        yy = y.reshape(-1, self.D)
        ly, _ = self.unmix(yy)
        ly = ly.reshape(y_shape)
        uu, logabsdet = self.spline.inverse(zz)
        logabsdet = torch.sum(logabsdet.reshape(z_shape[0],z_shape[1]), dim=1)
        u = uu.reshape(z_shape)
        sum_log_abs_det_jacobians += logabsdet
        lx, logabsdet = self.dconv.inverse(u, ly)
        sum_log_abs_det_jacobians += logabsdet
        lx = lx.reshape(-1, self.D)  
        x, logabsdet = self.unmix.inverse(lx)
        x = x.reshape(z_shape)
        logabsdet = torch.sum(logabsdet.reshape(z_shape[0],z_shape[1]), dim=1)
        sum_log_abs_det_jacobians += logabsdet
        return x, sum_log_abs_det_jacobians

    def log_prob(self, x, y):
         # z: [BS, T, D], e: [BS, T, D]
        e, z, sum_log_abs_det_jacobians = self.forward(x, y)
        logp = torch.sum(self.base_dist.log_prob(z), dim=1) + sum_log_abs_det_jacobians
        # TODO: density ratio trick to make e spatiotemporally independent
        return torch.mean(logp)        

    def sample(self, y, batch_size, length=8): 
        z = self.base_dist.sample((batch_size, length))
        # logp = self.base_dist.log_prob(z)
        x, _ = self.inverse(z, y)
        return x