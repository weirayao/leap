"""Temporal VAE with gaussian margial and laplacian transition prior"""

import torch
import numpy as np
import ipdb as pdb
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributions as D
from torch.nn import functional as F

from .components.beta import BetaVAE_MLP, BetaVAE_CNN
from .metrics.correlation import compute_mcc
from .components.base import GroupLinearLayer
from .components.transforms import ComponentWiseSpline


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(
            x_recon, x, size_average=False).div(batch_size)

    elif distribution == 'gaussian':
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

    elif distribution == 'sigmoid':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

    return recon_loss

def compute_cross_ent_normal(mu, logvar):
    return 0.5 * (mu**2 + torch.exp(logvar)) + np.log(np.sqrt(2 * np.pi))

def compute_ent_normal(logvar):
    return 0.5 * (logvar + np.log(2 * np.pi * np.e))

def compute_sparsity(mu, normed=True):
    # assume couples, compute normalized sparsity
    diff = mu[::2] - mu[1::2]
    if normed:
        norm = torch.norm(diff, dim=1, keepdim=True)
        norm[norm == 0] = 1  # keep those that are same, dont divide by 0
        diff = diff / norm
    return torch.mean(torch.abs(diff))


class AfflineVAESynthetic(pl.LightningModule):

    def __init__(
        self, 
        input_dim,
        z_dim=10, 
        lag=1,
        hidden_dim=128,
        beta=1,
        gamma=10,
        lr=1e-4,
        diagonal=False,
        decoder_dist='gaussian',
        rate_prior=1):
        '''Import Beta-VAE as encoder/decoder'''
        super().__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        
        self.net = BetaVAE_MLP(input_dim=input_dim, 
                               z_dim=z_dim, 
                               hidden_dim=hidden_dim)

        self.trans_func = GroupLinearLayer(din=_dim, 
                                           dout=z_dim,
                                           num_blocks=lag,
                                           diagonal=diagonal)

        self.b = nn.Parameter(0.01 * torch.randn(1, z_dim))

        self.spline = ComponentWiseSpline(input_dim=z_dim,
                                          bound=5,
                                          count_bins=8,
                                          order="linear")
        self.spline.load_state_dict(torch.load("/home/cmu_wyao/spline.pth"))
        self.lr = lr
        self.lag = lag
        self.beta = beta
        self.gamma = gamma
        self.rate_prior = rate_prior
        self.decoder_dist = decoder_dist

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(self.z_dim))
        self.register_buffer('base_dist_var', torch.eye(self.z_dim))

    @property
    def base_dist(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def forward(self, batch):
        xt, xt_ = batch["xt"], batch["xt_"]
        batch_size, _, _  = xt.shape
        x = torch.cat((xt, xt_), dim=1)
        x = x.view(-1, self.input_dim)
        return self.net(x)

    def compute_cross_ent_laplace(self, mean, logvar, rate_prior):
        var = torch.exp(logvar)
        sigma = torch.sqrt(var)
        ce = - torch.log(rate_prior / 2) + rate_prior * sigma *\
             np.sqrt(2 / np.pi) * torch.exp(- mean**2 / (2 * var)) -\
             rate_prior * mean * (
                     1 - 2 * self.normal_dist.cdf(mean / sigma))
        return ce

    def training_step(self, batch, batch_idx):
        xt, xt_ = batch["xt"], batch["xt_"]
        batch_size, _, _  = xt.shape
        x = torch.cat((xt, xt_), dim=1)
        x = x.view(-1, self.input_dim)
        x_recon, mu, logvar, z = self.net(x)
        
        # Normal VAE loss: recon_loss + kld_loss
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
        mu = mu.view(batch_size, -1, self.z_dim)
        logvar = logvar.view(batch_size, -1, self.z_dim)
        z = z.view(batch_size, -1, self.z_dim)
        mut, mut_ = mu[:,:-1,:], mu[:,-1:,:]
        logvart, logvart_ = logvar[:,:-1,:], logvar[:,-1:,:]
        zt, zt_ = z[:,:-1,:], z[:,-1:,:]
        
        # Past KLD divergenve
        p1 = D.Normal(torch.zeros_like(mut), torch.ones_like(logvart))
        q1 = D.Normal(mut, torch.exp(logvart / 2))
        log_qz_normal = q1.log_prob(zt)
        log_pz_normal = p1.log_prob(zt)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = torch.sum(torch.sum(kld_normal,dim=-1),dim=-1).mean()      
        
        # Current KLD divergence
        ut = self.trans_func(zt)
        ut = torch.sum(ut, dim=1) + self.b
        epst_ = zt_.squeeze() + ut
        et_, logabsdet = self.spline(epst_)
        log_pz_laplace = self.base_dist.log_prob(et_) + logabsdet
        q_laplace = D.Normal(mut_, torch.exp(logvart_ / 2))
        log_qz_laplace = q_laplace.log_prob(zt_)
        kld_laplace = torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace
        kld_laplace = kld_laplace.mean()       
        
        loss = (self.lag+1) * recon_loss + self.beta * kld_normal + self.gamma * kld_laplace  

        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_normal", kld_normal)
        self.log("train_kld_laplace", kld_laplace)

        return loss
    
    def validation_step(self, batch, batch_idx):
        xt, xt_ = batch["xt"], batch["xt_"]
        batch_size, _, _  = xt.shape
        x = torch.cat((xt, xt_), dim=1)
        x = x.view(-1, self.input_dim)
        x_recon, mu, logvar, z = self.net(x)
        
        # Normal VAE loss: recon_loss + kld_loss
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
        mu = mu.view(batch_size, -1, self.z_dim)
        logvar = logvar.view(batch_size, -1, self.z_dim)
        z = z.view(batch_size, -1, self.z_dim)
        mut, mut_ = mu[:,:-1,:], mu[:,-1:,:]
        logvart, logvart_ = logvar[:,:-1,:], logvar[:,-1:,:]
        zt, zt_ = z[:,:-1,:], z[:,-1:,:]
        
        # Past KLD divergenve
        p1 = D.Normal(torch.zeros_like(mut), torch.ones_like(logvart))
        q1 = D.Normal(mut, torch.exp(logvart / 2))
        log_qz_normal = q1.log_prob(zt)
        log_pz_normal = p1.log_prob(zt)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = torch.sum(torch.sum(kld_normal,dim=-1),dim=-1).mean()      
        
        # Current KLD divergence
        ut = self.trans_func(zt)
        ut = torch.sum(ut, dim=1) + self.b
        epst_ = zt_.squeeze() + ut
        et_, logabsdet = self.spline(epst_)
        log_pz_laplace = self.base_dist.log_prob(et_) + logabsdet
        q_laplace = D.Normal(mut_, torch.exp(logvart_ / 2))
        log_qz_laplace = q_laplace.log_prob(zt_)
        kld_laplace = torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace
        kld_laplace = kld_laplace.mean()

        loss = (self.lag+1) * recon_loss + self.beta * kld_normal + self.gamma * kld_laplace 

        zt_recon = mu[:,-1,:].T.detach().cpu().numpy()
        zt_true = batch["yt_"].squeeze().T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, "Pearson")
        self.log("val_mcc", mcc) 
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

    def reconstruct(self):
        return self.forward(batch)[0]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        return optimizer

class AfflineVAEKittiMask(pl.LightningModule):


    def __init__(
        self, 
        input_dim,
        z_dim=10, 
        lag = 1,
        hidden_dim = 128,
        beta = 1,
        gamma = 10,
        lr = 1e-4,
        diagonal = False,
        decoder_dist = 'gaussian',
        warm_start = False,
        rate_prior = 1):
        '''Import Beta-VAE as encoder/decoder'''
        super().__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        
        self.net = BetaVAE_CNN(input_dim=input_dim, 
                               z_dim=z_dim, 
                               hidden_dim=hidden_dim)

        self.trans_func = GroupLinearLayer(din = z_dim, 
                                           dout = z_dim,
                                           num_blocks = lag,
                                           diagonal = diagonal)

        self.b = nn.Parameter(0.01 * torch.randn(1, z_dim))

        self.spline = ComponentWiseSpline(input_dim = z_dim,
                                          bound = 5,
                                          count_bins = 8,
                                          order = "linear")
        if warm_start:
            self.spline.load_state_dict(torch.load("/home/cmu_wyao/spline.pth"))
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.lag = lag
        self.decoder_dist = decoder_dist
        self.rate_prior = rate_prior

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(self.z_dim))
        self.register_buffer('base_dist_var', torch.eye(self.z_dim))

    @property
    def base_dist(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def forward(self, batch):
        xt, xt_ = batch["xt"], batch["xt_"]
        batch_size, _, _  = xt.shape
        x = torch.cat((xt, xt_), dim=1)
        x = x.view(-1, self.input_dim)
        return self.net(x)

    def compute_cross_ent_laplace(self, mean, logvar, rate_prior):
        var = torch.exp(logvar)
        sigma = torch.sqrt(var)
        ce = - torch.log(rate_prior / 2) + rate_prior * sigma *\
             np.sqrt(2 / np.pi) * torch.exp(- mean**2 / (2 * var)) -\
             rate_prior * mean * (
                     1 - 2 * self.normal_dist.cdf(mean / sigma))
        return ce

    def training_step(self, batch, batch_idx):
        xt, xt_ = batch["xt"], batch["xt_"]
        batch_size, _, _  = xt.shape
        x = torch.cat((xt, xt_), dim=1)
        x = x.view(-1, self.input_dim)
        x_recon, mu, logvar, z = self.net(x)
        # Normal VAE loss: recon_loss + kld_loss
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
        mu = mu.view(batch_size, -1, self.z_dim)
        logvar = logvar.view(batch_size, -1, self.z_dim)
        z = z.view(batch_size, -1, self.z_dim)
        mut, mut_ = mu[:,:-1,:], mu[:,-1:,:]
        logvart, logvart_ = logvar[:,:-1,:], logvar[:,-1:,:]
        zt, zt_ = z[:,:-1,:], z[:,-1:,:]
        # Past KLD divergenve
        p1 = D.Normal(torch.zeros_like(mut), torch.ones_like(logvart))
        q1 = D.Normal(mut, torch.exp(logvart / 2))
        log_qz_normal = q1.log_prob(zt)
        log_pz_normal = p1.log_prob(zt)
        kld_normal = log_qz_normal - log_pz_normal

        kld_normal = torch.sum(torch.sum(kld_normal,dim=-1),dim=-1).mean()      
        # Current KLD divergence
        ut = self.trans_func(zt)
        ut = torch.sum(ut, dim=1) + self.b
        epst_ = zt_.squeeze() + ut
        et_, logabsdet = self.spline(epst_)
        log_pz_laplace = self.base_dist.log_prob(et_) + logabsdet
        q_laplace = D.Normal(mut_, torch.exp(logvart_ / 2))

        log_qz_laplace = q_laplace.log_prob(zt_)

        kld_laplace = torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace
        kld_laplace = kld_laplace.mean()       

        loss = (self.lag+1) * recon_loss + self.beta * kld_normal + self.gamma * kld_laplace  

        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", (self.lag+1) * recon_loss)
        self.log("train_kld_normal", kld_normal)
        self.log("train_kld_laplace", kld_laplace)

        return loss
    
    def validation_step(self, batch, batch_idx):
        xt, xt_ = batch["xt"], batch["xt_"]
        batch_size, _, _  = xt.shape
        x = torch.cat((xt, xt_), dim=1)
        x = x.view(-1, self.input_dim)
        x_recon, mu, logvar, z = self.net(x)
        # Normal VAE loss: recon_loss + kld_loss
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
        mu = mu.view(batch_size, -1, self.z_dim)
        logvar = logvar.view(batch_size, -1, self.z_dim)
        z = z.view(batch_size, -1, self.z_dim)
        mut, mut_ = mu[:,:-1,:], mu[:,-1:,:]
        logvart, logvart_ = logvar[:,:-1,:], logvar[:,-1:,:]
        zt, zt_ = z[:,:-1,:], z[:,-1:,:]
        # Past KLD divergenve
        p1 = D.Normal(torch.zeros_like(mut), torch.ones_like(logvart))
        q1 = D.Normal(mut, torch.exp(logvart / 2))
        log_qz_normal = q1.log_prob(zt)
        log_pz_normal = p1.log_prob(zt)
        kld_normal = log_qz_normal - log_pz_normal

        kld_normal = torch.sum(torch.sum(kld_normal,dim=-1),dim=-1).mean()      
        # Current KLD divergence
        ut = self.trans_func(zt)
        ut = torch.sum(ut, dim=1) + self.b
        epst_ = zt_.squeeze() + ut
        et_, logabsdet = self.spline(epst_)
        log_pz_laplace = self.base_dist.log_prob(et_) + logabsdet
        q_laplace = D.Normal(mut_, torch.exp(logvart_ / 2))

        log_qz_laplace = q_laplace.log_prob(zt_)

        kld_laplace = torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace
        kld_laplace = kld_laplace.mean()

        loss = (self.lag+1) * recon_loss + self.beta * kld_normal + self.gamma * kld_laplace 

        zt_recon = mu[:,-1,:].T.detach().cpu().numpy()
        zt_true = batch["yt_"].squeeze().T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, "Pearson")
        self.log("val_elbo_loss", loss)
        self.log("val_mcc", mcc) 
        self.log("val_elbo_loss", loss)
        self.log("val_recon_loss", (self.lag+1) * recon_loss)
        self.log("val_kld_normal", kld_normal)
        self.log("val_kld_laplace", kld_laplace)
        return loss
    
    def sample(self, xt):
        batch_size = xt.shape[0]
        e = torch.randn(batch_size, self.z_dim).to(xt.device)
        eps, _ = self.spline.inverse(e)
        return eps

    def reconstruct(self):
        return self.forward(batch)[0]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        # An scheduler is optional, but can help in flows to get the last bpd improvement
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return optimizer
