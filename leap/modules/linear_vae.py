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
        lr=1e-4,
        diagonal=False,
        bias=False,
        use_warm_start=False,
        spline_pth=None,
        decoder_dist='gaussian',
        correlation='Pearson'):
        '''Import Beta-VAE as encoder/decoder'''
        super().__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        
        self.net = BetaVAE_MLP(input_dim=input_dim, 
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
        self.correlation = correlation
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

    def training_step(self, batch, batch_idx):
        xt, xt_ = batch["xt"], batch["xt_"]
        batch_size, _, _  = xt.shape
        x = torch.cat((xt, xt_), dim=1)
        x = x.view(-1, self.input_dim)
        x_recon, mu, logvar, z = self.net(x)
        
        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
        mu = mu.view(batch_size, -1, self.z_dim)
        logvar = logvar.view(batch_size, -1, self.z_dim)
        z = z.view(batch_size, -1, self.z_dim)
        mut, mut_ = mu[:,:-1,:], mu[:,-1:,:]
        logvart, logvart_ = logvar[:,:-1,:], logvar[:,-1:,:]
        zt, zt_ = z[:,:-1,:], z[:,-1:,:]
        
        # Past KLD divergenve DKL[q(z_t-tau|x_t-tau) || p(z_t-tau)]
        p1 = D.Normal(torch.zeros_like(mut), torch.ones_like(logvart))
        q1 = D.Normal(mut, torch.exp(logvart / 2))
        log_qz_normal = q1.log_prob(zt)
        log_pz_normal = p1.log_prob(zt)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = torch.sum(torch.sum(kld_normal,dim=-1),dim=-1).mean()      
        
        # Current KLD divergence DKL[q(z_t|x_t) || p(z_t|{z_t-tau})]
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
        
        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
        mu = mu.view(batch_size, -1, self.z_dim)
        logvar = logvar.view(batch_size, -1, self.z_dim)
        z = z.view(batch_size, -1, self.z_dim)
        mut, mut_ = mu[:,:-1,:], mu[:,-1:,:]
        logvart, logvart_ = logvar[:,:-1,:], logvar[:,-1:,:]
        zt, zt_ = z[:,:-1,:], z[:,-1:,:]
        
        # Past KLD divergenve DKL[q(z_t-tau|x_t-tau) || p(z_t-tau)]
        p1 = D.Normal(torch.zeros_like(mut), torch.ones_like(logvart))
        q1 = D.Normal(mut, torch.exp(logvart / 2))
        log_qz_normal = q1.log_prob(zt)
        log_pz_normal = p1.log_prob(zt)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = torch.sum(torch.sum(kld_normal,dim=-1),dim=-1).mean()      
        
        # Current KLD divergence DKL[q(z_t|x_t) || p(z_t|{z_t-tau})]
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

        # Compute Mean Correlation Coefficient
        zt_recon = mu[:,-1,:].T.detach().cpu().numpy()
        zt_true = batch["yt_"].squeeze().T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)

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

class AfflineVAECNN(pl.LightningModule):

    def __init__(
        self, 
        nc,
        z_dim=10, 
        lag=1,
        hidden_dim=256,
        bound=5,
        count_bins=8,
        order='linear',
        beta=0.0025,
        gamma=0.0075,
        l1=0.0,
        lr=1e-4,
        diagonal=True,
        identity=False,
        bias=False,
        use_warm_start=False,
        spline_pth=None,
        decoder_dist='bernoulli',
        correlation='Pearson'):
        '''Import Beta-VAE as encoder/decoder'''
        super().__init__()
        self.z_dim = z_dim
        self.nc = nc
        
        self.net = BetaVAE_CNN(nc=nc, 
                               z_dim=z_dim,
                               hidden_dim=hidden_dim)

        self.trans_func = GroupLinearLayer(din=z_dim, 
                                           dout=z_dim,
                                           num_blocks=lag,
                                           diagonal=diagonal)
        if identity:
            # SlowVAE setting: identity transitions
            with torch.no_grad():
                self.trans_func.d.fill_(1.)
                # self.trans_func.d.requires_grad = False
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
        self.l1 = l1
        self.lag = lag
        self.beta = beta
        self.gamma = gamma
        self.correlation = correlation
        self.decoder_dist = decoder_dist

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(self.z_dim))
        self.register_buffer('base_dist_var', torch.eye(self.z_dim))

    @property
    def base_dist(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def forward(self, batch):
        xt, xt_ = batch["xt"], batch["xt_"]
        batch_size, _, nc, h, w  = xt.shape
        x = torch.cat((xt, xt_), dim=1)
        x = x.view(-1, nc, h, w)
        return self.net(x)

    def training_step(self, batch, batch_idx):
        xt, xt_ = batch["xt"], batch["xt_"]
        batch_size, _, nc, h, w  = xt.shape
        x = torch.cat((xt, xt_), dim=1)
        x = x.view(-1, nc, h, w)
        x_recon, mu, logvar, z = self.net(x)
        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
        mu = mu.view(batch_size, -1, self.z_dim)
        logvar = logvar.view(batch_size, -1, self.z_dim)
        z = z.view(batch_size, -1, self.z_dim)
        mut, mut_ = mu[:,:-1,:], mu[:,-1:,:]
        logvart, logvart_ = logvar[:,:-1,:], logvar[:,-1:,:]
        zt, zt_ = z[:,:-1,:], z[:,-1:,:]
        
        # Past KLD divergenve DKL[q(z_t-tau|x_t-tau) || p(z_t-tau)]
        p1 = D.Normal(torch.zeros_like(mut), torch.ones_like(logvart))
        q1 = D.Normal(mut, torch.exp(logvart / 2))
        log_qz_normal = q1.log_prob(zt)
        log_pz_normal = p1.log_prob(zt)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = torch.sum(torch.sum(kld_normal,dim=-1),dim=-1).mean()      
        
        # Current KLD divergence DKL[q(z_t|x_t) || p(z_t|{z_t-tau})]
        ut = self.trans_func(zt)
        ut = torch.sum(ut, dim=1) + self.b
        epst_ = zt_.squeeze() + ut
        et_, logabsdet = self.spline(epst_)
        log_pz_laplace = self.base_dist.log_prob(et_) + logabsdet
        q_laplace = D.Normal(mut_, torch.exp(logvart_ / 2))
        log_qz_laplace = q_laplace.log_prob(zt_)
        kld_laplace = torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace
        kld_laplace = kld_laplace.mean()       
        
        # L1 penalty to encourage sparcity in causal matrix
        l1_loss = 0
        for param in self.trans_func.parameters():
            l1_loss = l1_loss + torch.norm(param, 1)

        loss = (self.lag+1) * recon_loss + self.beta * kld_normal + self.gamma * kld_laplace + self.l1 * l1_loss

        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_normal", kld_normal)
        self.log("train_kld_laplace", kld_laplace)

        return loss
    
    def validation_step(self, batch, batch_idx):
        xt, xt_ = batch["xt"], batch["xt_"]
        batch_size, _, nc, h, w  = xt.shape
        x = torch.cat((xt, xt_), dim=1)
        x = x.view(-1, nc, h, w)
        x_recon, mu, logvar, z = self.net(x)
        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
        mu = mu.view(batch_size, -1, self.z_dim)
        logvar = logvar.view(batch_size, -1, self.z_dim)
        z = z.view(batch_size, -1, self.z_dim)
        mut, mut_ = mu[:,:-1,:], mu[:,-1:,:]
        logvart, logvart_ = logvar[:,:-1,:], logvar[:,-1:,:]
        zt, zt_ = z[:,:-1,:], z[:,-1:,:]
        
        # Past KLD divergenve DKL[q(z_t-tau|x_t-tau) || p(z_t-tau)]
        p1 = D.Normal(torch.zeros_like(mut), torch.ones_like(logvart))
        q1 = D.Normal(mut, torch.exp(logvart / 2))
        log_qz_normal = q1.log_prob(zt)
        log_pz_normal = p1.log_prob(zt)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = torch.sum(torch.sum(kld_normal,dim=-1),dim=-1).mean()      
        
        # Current KLD divergence DKL[q(z_t|x_t) || p(z_t|{z_t-tau})]
        ut = self.trans_func(zt)
        ut = torch.sum(ut, dim=1) + self.b
        epst_ = zt_.squeeze() + ut
        et_, logabsdet = self.spline(epst_)
        log_pz_laplace = self.base_dist.log_prob(et_) + logabsdet
        q_laplace = D.Normal(mut_, torch.exp(logvart_ / 2))
        log_qz_laplace = q_laplace.log_prob(zt_)
        kld_laplace = torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace
        kld_laplace = kld_laplace.mean()       
        
        # L1 penalty to encourage sparcity in causal matrix
        l1_loss = 0
        for param in self.trans_func.parameters():
            l1_loss = l1_loss + torch.norm(param, 1)

        loss = (self.lag+1) * recon_loss + self.beta * kld_normal + self.gamma * kld_laplace + self.l1 * l1_loss
        # Compute Mean Correlation Coefficient
        zt_recon = mu[:,-1,:].T.detach().cpu().numpy()
        zt_true = batch["yt_"].squeeze().T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        # An scheduler is optional, but can help in flows to get the last bpd improvement
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.9)
        return [optimizer], [scheduler]