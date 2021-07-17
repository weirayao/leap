"""Temporal VAE with gaussian margial and laplacian transition prior"""

import torch
import numpy as np
import ipdb as pdb
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributions as D
from torch.autograd import Variable
from torch.nn import functional as F

from .metrics.correlation import compute_mcc
from .components.base import GroupLinearLayer
from .components.transforms import ComponentWiseSpline
from .components.vae_nonlinear_gau_ts import BetaVAE_MLP_ts


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(
            x_recon, x, size_average=False).div(batch_size)

    elif distribution == 'gaussian':
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

    elif distribution == 'sigmoid_gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

    return recon_loss

class AfflineVAESynthetic(pl.LightningModule):

    def __init__(
        self, 
        input_dim,
        z_dim=10, 
        lag=1,
        hidden_dim=128,
        bound=5,
        count_bins=8,
        order='linear',
        beta=0.0025,
        gamma=0.0075,
        alpha=0.0075,
        lr=1e-4,
        diagonal=False,
        bias=False,
        use_warm_start=False,
        spline_pth=None,
        decoder_dist='gaussian',
        correlation='Pearson'):
        super().__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        
        self.net = BetaVAE_MLP_ts(input_dim=input_dim, 
                               z_dim=z_dim, 
                               hidden_dim=hidden_dim)

        self.trans_func = GroupLinearLayer(din=z_dim, 
                                           dout=z_dim,
                                           num_blocks=lag,
                                           diagonal=diagonal)

        # Non-white noise: use learned bias to adjust
        if bias:
            self.b = nn.Parameter(0.001 * torch.randn(1, z_dim))
        else:
            self.register_buffer('b', torch.zeros((1, z_dim)))

        self.spline = ComponentWiseSpline(input_dim=z_dim,
                                          bound=bound,
                                          count_bins=count_bins,
                                          order=order)
        if use_warm_start:
            self.spline.load_state_dict(torch.load(spline_pth, 
                                        map_location=torch.device('cpu')))
            print("Load pretrained spline flow", flush=True)
        self.lr = lr
        self.lag = lag
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.correlation = correlation
        self.decoder_dist = decoder_dist
        self.prior_rnn = nn.LSTM(z_dim*2, z_dim, num_layers=1, batch_first=True)

        # Beta
        self.mu_layer = nn.Linear(z_dim, z_dim)
        self.sigma_layer = nn.Linear(z_dim, z_dim)

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_var', torch.eye(self.z_dim))
        self.register_buffer('base_dist_mean', torch.zeros(self.z_dim))

    @property
    def base_dist(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def forward(self, batch):
        xt, yt = batch["xt"], batch["yt"]
        xt = xt.to(device); yt = yt.to(device)
        return self.net(xt, self.lag)

    def training_step(self, batch, batch_idx):
        xt, yt = batch["xt"], batch["yt"]
        batch_size, length, latent_dim  = xt.shape
        x_recon, mut, logvart, latent = self.net(xt, self.lag)
        
        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = reconstruction_loss(xt.reshape(-1, latent_dim), x_recon.reshape(-1, latent_dim), self.decoder_dist)

        # Past KLD divergenve DKL[q(z_t-tau|x_t-tau) || p(z_t-tau)]
        z_init = latent[:, 0:self.lag, :]
        mut_init = mut[:, 0:self.lag, :]
        logvart_init = logvart[:, 0:self.lag, :]
        
        p1 = D.Normal(torch.zeros_like(mut_init), torch.ones_like(logvart_init))
        q1 = D.Normal(mut_init, torch.exp(logvart_init / 2))
        log_qz_normal = q1.log_prob(z_init)
        log_pz_normal = p1.log_prob(z_init)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = torch.abs(torch.sum(torch.sum(kld_normal,dim=-1),dim=-1).mean())      
        
        # Current KLD divergence DKL[q(z_t|x_t) || p(z_t|{z_t-tau})]
        kld_laplace = 0
        for i in range(length-self.lag):
            zt = latent[:,i:i+self.lag,:]
            zt_ = latent[:,i+self.lag,:]

            mut_ = mut[:,i+self.lag,:]
            logvart_ = logvart[:,i+self.lag,:]

            ut = self.trans_func(zt)
            ut = torch.sum(ut, dim=1) + self.b
            epst_ = zt_ + ut
            et_, logabsdet = self.spline(epst_)
            log_pz_laplace = self.base_dist.log_prob(et_) + logabsdet
            q_laplace = D.Normal(mut_, torch.exp(logvart_ / 2))
            log_qz_laplace = q_laplace.log_prob(zt_)
            kld_laplace += torch.abs((torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace).mean())

        loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace

        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_normal", kld_normal)
        self.log("train_kld_laplace", kld_laplace)

        return loss
    
    def validation_step(self, batch, batch_idx):
        xt, yt = batch["xt"], batch["yt"]
        batch_size, length, latent_dim  = xt.shape
        x_recon, mut, logvart, latent = self.net(xt, self.lag)
        
        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = reconstruction_loss(xt.reshape(-1, latent_dim), x_recon.reshape(-1, latent_dim), self.decoder_dist)

        # Past KLD divergenve DKL[q(z_t-tau|x_t-tau) || p(z_t-tau)]
        z_init = latent[:, 0:self.lag, :]
        mut_init = mut[:, 0:self.lag, :]
        logvart_init = logvart[:, 0:self.lag, :]
        
        p1 = D.Normal(torch.zeros_like(mut_init), torch.ones_like(logvart_init))
        q1 = D.Normal(mut_init, torch.exp(logvart_init / 2))
        log_qz_normal = q1.log_prob(z_init)
        log_pz_normal = p1.log_prob(z_init)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = torch.abs(torch.sum(torch.sum(kld_normal,dim=-1),dim=-1).mean())      
        
        # Current KLD divergence DKL[q(z_t|x_t) || p(z_t|{z_t-tau})]
        kld_laplace = 0
        for i in range(length-self.lag):
            zt = latent[:,i:i+self.lag,:]
            zt_ = latent[:,i+self.lag,:]

            mut_ = mut[:,i+self.lag,:]
            logvart_ = logvart[:,i+self.lag,:]

            ut = self.trans_func(zt)
            ut = torch.sum(ut, dim=1) + self.b
            epst_ = zt_ + ut
            et_, logabsdet = self.spline(epst_)
            log_pz_laplace = self.base_dist.log_prob(et_) + logabsdet
            q_laplace = D.Normal(mut_, torch.exp(logvart_ / 2))
            log_qz_laplace = q_laplace.log_prob(zt_)
            kld_laplace += torch.abs((torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace).mean())

        loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace
        # Compute Mean Correlation Coefficient
        mcc = 0
        for i in range(length):
            zt_recon = mut[:, i, :].T.detach().cpu().numpy()
            zt_true = yt[:, i, :].T.detach().cpu().numpy()
            mcc += compute_mcc(zt_recon, zt_true, self.correlation)

        self.log("val_mcc", mcc/length) 
        self.log("val_elbo_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kld_normal", kld_normal)
        self.log("val_kld_laplace", kld_laplace)

        return loss
    
    def sample(self, xt):
        batch_size = xt.shape[0]
        e = torch.randn(batch_size, self.z_dim).to(xt.device)
        eps, _ = self.spline.inverse(e)
        return eps

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        return optimizer
