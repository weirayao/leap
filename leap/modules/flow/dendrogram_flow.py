import torch
import torch.nn as nn
from torch.nn import functional as F, init
import torch.distributions as D
from .components.glow import Glow
from .components.base import GroupLinearLayer
from .components.transforms import ComponentWiseSpline
import ipdb as pdb

class AfflineDendrogramFlow(nn.Module):
    """Dendrogram flow to process/generate videos/next frame with affine transitions"""
    def __init__(
        self,
        input_dims=(3,64,64),
        width = 64,
        depth = 16,
        n_levels = 3,
        lags = 4):
        super().__init__()
        self.input_dims = input_dims
        channels, H, W = input_dims
        self.P = lags
        self.L = n_levels
        self.D = input_dims
        self._compute_zs_dims()
        self.unmix = Glow(width=width, depth=depth, n_levels=n_levels, 
                          input_dims=input_dims, checkpoint_grads=False)
        self.spline = ComponentWiseSpline(input_dim = channels*H*W,
                                          bound = 8,
                                          count_bins = 8,
                                          order = "linear")
        # Initialize linear transition kernels
        topdown_kernel = { }
        topdown_bias = { }
        # Note we remove the transitions in bottom layers
        # So they can encode irrelevant noise (n << d)
        # Top level is level 0, the second last level is self.L-1
        for past_level in range(self.L):
            for curr_level in range(past_level, self.L):
                channels, H, W = self.zs_dims[past_level]
                din = channels * H * W
                channels, H, W = self.zs_dims[curr_level]
                dout = channels * H * W
                topdown_kernel["L%d-L%d"%(past_level,curr_level)] = GroupLinearLayer(din, dout, self.P, False, 8)
                topdown_bias["L%d-L%d"%(past_level,curr_level)] = nn.Parameter(0.01 * torch.randn(1, dout))
        self.topdown_kernel = nn.ModuleDict(topdown_kernel)
        self.topdown_bias = nn.ParameterDict(topdown_bias)
        # base distribution of the flow
        self.register_buffer('base_dist_mean', torch.zeros(1))
        self.register_buffer('base_dist_var', torch.ones(1))

    def forward(self, x, u):
        # x: [BS, C, H, W], u: [BS, P, C, H, W]
        BS, C, H, W = x.shape
        _, P, _,_,_ = u.shape
        uu = u.reshape(BS*P, C, H, W)
        sum_log_abs_det_jacobians = 0
        # Note GLOW lu/lx[0] is the bottom level
        # D = CxHxW         
        # NOW REVERSE THE ORDER SO LEVEL 0 IS TOP LEVEL in xx
        lu, _ = self.unmix(uu)
        lu = [z.reshape(BS, P, -1) for z in lu]
        lu.reverse()
        lx, logabsdet = self.unmix(x)
        lx = [z.reshape(BS, -1) for z in lx]
        lx.reverse()
        sum_log_abs_det_jacobians += logabsdet
        # Recover noise from transition
        es = [ ]
        for curr_level in reversed(range(self.L)):
            cond = 0
            for past_level in range(curr_level+1):
                ut = self.topdown_kernel["L%d-L%d"%(past_level,curr_level)](lu[past_level])
                ut = torch.sum(ut, dim=1) + self.topdown_bias["L%d-L%d"%(past_level,curr_level)]
                cond = cond + ut
            u0, logabsdet = lx[curr_level], torch.zeros(BS).to(x.device)
            e = cond + u0
            es.append(e)
            sum_log_abs_det_jacobians += logabsdet
        es.reverse()
        # Bottom layer assumes no transition, directly "gaussianize/laplacianize"
        e, logabsdet = lx[self.L], torch.zeros(BS).to(x.device)
        es.append(e)
        sum_log_abs_det_jacobians += logabsdet
        # Spline flow
        ee = torch.cat(es, dim=1)
        z, logabsdet = self.spline(ee)
        sum_log_abs_det_jacobians += logabsdet
        return es, z, sum_log_abs_det_jacobians
    
    def inverse(self, z, u):
        BS, P, C, H, W = u.shape
        uu = u.reshape(BS*P, C, H, W)
        lu, _ = self.unmix(uu)
        lu = [z.reshape(BS, P, -1) for z in lu]
        lu.reverse()
        sum_log_abs_det_jacobians = 0
        # Spline flow
        ee, logabsdet = self.spline.inverse(z)
        sum_log_abs_det_jacobians += logabsdet
        es = [ ]
        start_dim = 0
        for level in range(len(self.zs_dims)):
            channels, H, W  = self.zs_dims[level]
            dims = channels*H*W
            end_dim = start_dim + dims
            es.append(ee[:,start_dim:end_dim])
            start_dim = end_dim
        # Generate lx
        lx = [ ]
        for curr_level in reversed(range(self.L)):
            cond = 0
            for past_level in range(curr_level+1):
                ut = self.topdown_kernel["L%d-L%d"%(past_level,curr_level)](lu[past_level])
                ut = torch.sum(ut, dim=1) + self.topdown_bias["L%d-L%d"%(past_level,curr_level)]
                cond = cond + ut
            e = es[curr_level]
            u0 = e - cond
            u0 = u0.reshape(BS, self.zs_dims[curr_level][0], self.zs_dims[curr_level][1], self.zs_dims[curr_level][2])
            lx.append(u0)
            logabsdet = torch.zeros(BS).to(u.device)
            sum_log_abs_det_jacobians += logabsdet
        lx.reverse()
        # Bottom layer assumes no transition, directly "gaussianize/laplacianize"
        u0, logabsdet = es[self.L], torch.zeros(BS).to(u.device)
        u0 = u0.reshape(BS, self.zs_dims[self.L][0], self.zs_dims[self.L][1], self.zs_dims[self.L][2])
        lx.append(u0)
        sum_log_abs_det_jacobians += logabsdet
        lx.reverse()
        x, logabsdet = self.unmix.inverse(lx)
        sum_log_abs_det_jacobians += logabsdet
        return x, sum_log_abs_det_jacobians

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def log_prob(self, x, u):
        _, z, logdet = self.forward(x, u)
        log_prob = self.base_dist.log_prob(z).sum(dim=1) + logdet
        return log_prob

    def _compute_zs_dims(self):
        self.zs_dims = [ ]
        channels, H, W = self.input_dims
        n_levels = self.L
        for level in range(n_levels):
            channels = channels * 2
            H = H //2
            W = W //2
            latent_dims = torch.Size((channels, H, W))
            self.zs_dims.append(latent_dims)
        in_channels, H, W = self.input_dims
        out_channels = int(in_channels * 4**(n_levels+1) / 2**n_levels)
        out_HW = int(H / 2**(n_levels+1))                         
        output_dims = out_channels, out_HW, out_HW
        self.zs_dims.append(torch.Size(output_dims))
        self.zs_dims.reverse()
    
    def inference(self, x):
        lx, logabsdet = self.unmix(x)
        lx.reverse()
        return lx

    def generate_next_frame(self, u, batch_size):
        channels, H, W = self.input_dims
        dims = channels*H*W 
        z = 0.1*self.base_dist.sample((batch_size, dims)).squeeze()
        x, _ = self.inverse(z, u)
        return x
        
    def sample_noise(self, batch_size):
        channels, H, W = self.input_dims
        dims = channels*H*W 
        with torch.no_grad():
            z = self.base_dist.sample((batch_size, dims)).squeeze()
            u, _ = self.spline.inverse(z)
            return u