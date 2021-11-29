import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from .base import GroupLinearLayer
from .transforms import ComponentWiseSpline
import ipdb as pdb

class Actnorm(nn.Module):
    """ Actnorm layer; cf Glow section 3.1 """
    def __init__(self, param_dim=(1,2)):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(param_dim))
        self.bias = nn.Parameter(torch.zeros(param_dim) + 5)

    def forward(self, x):
        z = (x - self.bias) / self.scale
        return z

class LinearTemporalVAESynthetic(nn.Module):


    def __init__(
        self, 
        input_dim,
        y_dim, 
        lag = 1,
        e_dim = 128,
        diagonal = False,
        kl_coeff = 1,
        negative_slope = 0.2):
        '''
        please add your flow module
        self.flow = XX
        '''
        super().__init__()
        self.kl_coeff = kl_coeff
        self.e_dim = e_dim
        self.y_dim = y_dim
        self.input_dim = input_dim
        self.lag = lag 
        self.unmix_func = nn.Sequential(
                                       nn.Linear(input_dim, e_dim),
                                       nn.BatchNorm1d(e_dim),
                                       nn.ReLU(negative_slope),
                                       nn.Linear(e_dim, e_dim),
                                       nn.BatchNorm1d(e_dim),
                                       nn.ReLU(negative_slope),
                                       nn.Linear(e_dim, e_dim),
                                       nn.BatchNorm1d(e_dim),
                                       nn.ReLU(negative_slope),
                                       nn.Linear(e_dim, y_dim)
                                       )

        self.mix_func = nn.Sequential(
                                       nn.Linear(y_dim, e_dim),
                                       nn.BatchNorm1d(e_dim),
                                       nn.ReLU(negative_slope),
                                       nn.Linear(e_dim, e_dim),
                                       nn.BatchNorm1d(e_dim),
                                       nn.ReLU(negative_slope),
                                       nn.Linear(e_dim, e_dim),
                                       nn.BatchNorm1d(e_dim),
                                       nn.ReLU(negative_slope),
                                       nn.Linear(e_dim, input_dim)
                                       )

        self.trans_func = GroupLinearLayer(din = y_dim, 
                                           dout = y_dim,
                                           num_blocks = lag,
                                           diagonal = diagonal)

        self.b = nn.Parameter(0.01 * torch.randn(1, y_dim))

        self.spline = ComponentWiseSpline(input_dim = y_dim,
                                          bound = 8,
                                          count_bins = 8,
                                          order = "linear")
        self.spline.load_state_dict(torch.load("/home/cmu_wyao/spline.pth"))
        # for param in self.spline.parameters():
        #     param.requires_grad = False
        self.e_mean = nn.Sequential(
                                     nn.Linear(y_dim, e_dim),
                                     nn.ReLU(negative_slope),
                                     nn.Linear(e_dim, e_dim),
                                     nn.ReLU(negative_slope),
                                     nn.Linear(e_dim, y_dim)
                                     )  

        self.e_logvar = nn.Sequential(
                                     nn.Linear(y_dim, e_dim),
                                     nn.ReLU(negative_slope),
                                     nn.Linear(e_dim, e_dim),
                                     nn.ReLU(negative_slope),
                                     nn.Linear(e_dim, y_dim)
                                     )                                     

    def encode(self, batch):
        xt, xt_ = batch["xt"], batch["xt_"]
        batch_size, _, _  = xt.shape
        input_x = torch.cat((xt, xt_), dim=1)
        input_x = input_x.view(-1, self.input_dim)
        y = self.unmix_func(input_x)
        y = y.view(batch_size, self.lag+1, self.y_dim)
        yt, yt_ = y[:, :-1, :], y[:, -1:, :]
        eps_t, ut = self._compute_noise(yt, yt_)
        e_mean = self.e_mean(eps_t)
        e_logvar = self.e_logvar(eps_t)
        # e_logvar = torch.zeros_like(e_mean)  - 10
        return  e_mean, e_logvar, yt, yt_, ut

    def decode(self, eps, ut):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        yt_ = ut + eps
        xt_ = self.mix_func(yt_)
        return xt_.unsqueeze(1)

    def forward(self, batch):
        # Past frames/snapshots: xt = x_t           (batch_size, length, size)
        # Current frame/snapshot: xt_ = x_(t+1)     (batch_size, length, size)
        e_mean, e_logvar, yt, yt_, ut = self.encode(batch)
        p, q, e = self.reparameterize(e_mean, e_logvar)
        xt_ = batch["xt_"]
        eps, _ = self.spline.inverse(e)
        recon_xt_ =  self.decode(eps, ut)
        return e_mean, e_logvar, yt, yt_, xt_, recon_xt_, p, q, e, eps

    def elbo_loss(self, x, x_hat, p, q, z):
        recon_loss = torch.square(x_hat - x).squeeze().sum(dim=1).mean()
        # recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)
        kl = log_qz - log_pz
        kl = kl.mean()
        loss = self.kl_coeff * kl + recon_loss
        return loss
        
    def predict_next_latent(self, xt):
        xt = xt.view(-1, self.input_dim)
        yt = self.unmix_func(xt)
        yt = yt.view(-1, self.lag, self.y_dim)
        ut = self.trans_func(yt)
        ut = torch.sum(ut, dim=1) + self.b
        return ut

    def _compute_noise(self, yt, yt_):
        """
        yt: Past snapshots [BS, lags, D]
        yt_: Current snapshot [BS, 1, D]
        """
        ut = self.trans_func(yt)
        ut = torch.sum(ut, dim=1) + self.b
        eps_t = yt_.squeeze() - ut
        return eps_t, ut

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = D.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = D.Normal(mu, std)
        z = q.rsample()
        return p, q, z