"""Temporal VAE with gaussian margial and laplacian transition prior"""
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
from torch.nn import functional as F

import pytorch_lightning as pl

from .components.beta import BetaVAE_Physics
from .components.graph import GNNModel
from .metrics.correlation import compute_mcc
from .components.transforms import ComponentWiseSpline

import ipdb as pdb

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

class PhysicsVAE(pl.LightningModule):

    def __init__(
        self, 
        nc,
        n_obj,
        z_dim=10,
        lag=1, 
        hidden_dim=256,
        num_layers=2,
        layer_name='GAT',
        bound=5,
        count_bins=8,
        order='linear',
        beta=0.0025,
        gamma=0.0075,
        l1=0.0,
        lr=1e-4,
        use_warm_start=False,
        spline_pth=None,
        decoder_dist='gaussian',
        correlation='Spearman'):
        '''Import Beta-VAE as encoder/decoder'''
        super().__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.n_obj = n_obj
        self.lag = lag

        self.net = BetaVAE_Physics(z_dim=z_dim,
                                   n_obj=n_obj,
                                   nc=nc,
                                   hidden_dim=hidden_dim,
                                   height=64,
                                   width=64)

        self.trans_func = GNNModel(c_in=z_dim, 
                                   c_hidden=hidden_dim, 
                                   c_out=z_dim, 
                                   num_layers=num_layers, 
                                   layer_name=layer_name,
                                   dp_rate=0.0, 
                                   num_heads=2,
                                   alpha=0.2)

        self.spline = ComponentWiseSpline(input_dim=z_dim * n_obj,
                                          bound=bound,
                                          count_bins=count_bins,
                                          order=order)

        if use_warm_start:
            self.spline.load_state_dict(torch.load(spline_pth, 
                                        map_location=torch.device('cpu')))
            print("Load pretrained spline flow", flush=True)

        self.lr = lr
        self.l1 = l1
        self.beta = beta
        self.gamma = gamma
        self.correlation = correlation
        self.decoder_dist = decoder_dist
        
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(z_dim * n_obj))
        self.register_buffer('base_dist_var', torch.eye(z_dim * n_obj))

        # Adjacency matrix logits parametrized by logits
        logits = torch.randn(n_obj, n_obj, 2)
        # Initialized to fully-connected graphs
        logits[:,:,1] = logits[:,:,1] - 5
        self.logits = nn.Parameter(logits)
        
    @property
    def base_dist(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def forward(self, batch):
        x, kps_gt, graph_gt, actions = batch
        batch_size, steps, nc, h, w  = x.shape
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, kps_gt, graph_gt, actions = batch
        batch_size, steps, nc, h, w  = x.shape

        # Sample masked adjacent matrix [BS, n_obj, n_obj] from logits 
        adj_matrix = F.gumbel_softmax(self.logits, hard=True)[:,:,0].fill_diagonal_(1)
        adj_batch = adj_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

        x_recon, mu, logvar, z = self.net(x)

        # Concatenate x to the shape of x_recon triplets
        triples = []
        for i in range(steps - 2):
            # pair consecutive frames (n, 2c, w, h)
            triple = torch.stack((x[:, i], x[:, i+1], x[:, i+2]), 1)
            triples.append(triple)
        triples = torch.stack(triples, dim=1)

        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = reconstruction_loss(triples, x_recon, self.decoder_dist)
        mut, mut_ = mu[:,0], mu[:,1]
        logvart, logvart_ = logvar[:,0], logvar[:,1]
        zt, zt_ = z[:,0], z[:,1]

        # Past KLD divergenve DKL[q(z_t-tau|x_t-tau) || p(z_t-tau)]
        p1 = D.Normal(torch.zeros_like(mut), torch.ones_like(logvart))
        q1 = D.Normal(mut, torch.exp(logvart / 2))
        log_qz_normal = q1.log_prob(zt)
        log_pz_normal = p1.log_prob(zt)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = torch.sum(torch.sum(kld_normal,dim=-1),dim=-1).mean()      
        
        # Current KLD divergence DKL[q(z_t|x_t) || p(z_t|{z_t-tau})]
        ut = self.trans_func(zt, adj_batch) + zt # Skip connection
        epst_ = zt_.squeeze() + ut
        et_, logabsdet = self.spline(epst_.view(batch_size, -1))
        log_pz_laplace = self.base_dist.log_prob(et_) + logabsdet
        q_laplace = D.Normal(mut_, torch.exp(logvart_ / 2))
        log_qz_laplace = q_laplace.log_prob(zt_)
        kld_laplace = torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace
        kld_laplace = kld_laplace.mean()       
        
        # L1 penalty to encourage sparcity in causal matrix
        # Masked Gradient-Based Causal Structure Learning
        # https://arxiv.org/abs/1910.08527
        off_diag_mask = ~torch.eye(adj_matrix.shape[0],dtype=bool, device=adj_matrix.device)
        l1_loss = torch.norm(torch.masked_select(adj_matrix, off_diag_mask), 1)

        loss = (self.lag+1) * recon_loss + self.beta * kld_normal + self.gamma * kld_laplace + self.l1 * l1_loss

        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_normal", kld_normal)
        self.log("train_kld_laplace", kld_laplace)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, kps_gt, graph_gt, actions = batch
        batch_size, steps, nc, h, w  = x.shape

        # Sample masked adjacent matrix [BS, n_obj, n_obj] from logits 
        adj_matrix = F.gumbel_softmax(self.logits, hard=True)[:,:,0].fill_diagonal_(1)
        adj_batch = adj_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

        x_recon, mu, logvar, z = self.net(x)
        triples = []
        for i in range(steps - 2):
            # pair consecutive frames (n, 2c, w, h)
            triple = torch.stack((x[:, i], x[:, i+1], x[:, i+2]), 1)
            triples.append(triple)
        triples = torch.stack(triples, dim=1)

        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = reconstruction_loss(triples, x_recon, self.decoder_dist)
        mut, mut_ = mu[:,0], mu[:,1]
        logvart, logvart_ = logvar[:,0], logvar[:,1]
        zt, zt_ = z[:,0], z[:,1]

        # Past KLD divergenve DKL[q(z_t-tau|x_t-tau) || p(z_t-tau)]
        p1 = D.Normal(torch.zeros_like(mut), torch.ones_like(logvart))
        q1 = D.Normal(mut, torch.exp(logvart / 2))
        log_qz_normal = q1.log_prob(zt)
        log_pz_normal = p1.log_prob(zt)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = torch.sum(torch.sum(kld_normal,dim=-1),dim=-1).mean()      
        
        # Current KLD divergence DKL[q(z_t|x_t) || p(z_t|{z_t-tau})]
        ut = self.trans_func(zt, adj_batch) + zt # Skip connection
        epst_ = zt_.squeeze() + ut
        et_, logabsdet = self.spline(epst_.view(batch_size, -1))
        log_pz_laplace = self.base_dist.log_prob(et_) + logabsdet
        q_laplace = D.Normal(mut_, torch.exp(logvart_ / 2))
        log_qz_laplace = q_laplace.log_prob(zt_)
        kld_laplace = torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace
        kld_laplace = kld_laplace.mean()       
        
        # L1 penalty to encourage sparcity in causal matrix
        off_diag_mask = ~torch.eye(adj_matrix.shape[0],dtype=bool, device=adj_matrix.device)
        l1_loss = torch.norm(torch.masked_select(adj_matrix, off_diag_mask), 1)

        loss = (self.lag+1) * recon_loss + self.beta * kld_normal + self.gamma * kld_laplace + self.l1 * l1_loss

        # Compute Mean Correlation Coefficient
        zt_recon = mu[:,1].contiguous().view(batch_size, -1).T.detach().cpu().numpy()
        zt_true = kps_gt[:,1].contiguous().view(batch_size, -1).T.detach().cpu().numpy()
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # An scheduler is optional, but can help in flows to get the last bpd improvement
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.9)
        return [optimizer], [scheduler]