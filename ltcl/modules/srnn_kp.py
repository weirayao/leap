"""Temporal VAE with gaussian margial and laplacian transition prior"""

import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as transforms

from .metrics.correlation import compute_mcc
from .components.tc import Discriminator, permute_dims
from .components.transforms import ComponentWiseSpline
from .components.mlp import Inference
from .components.transition import (MBDTransitionPrior, 
                                    NPTransitionPrior)

from .keypointer import Keypointer

import ipdb as pdb

class SRNNKeypointNS(pl.LightningModule):

    def __init__(
        self, 
        nc,
        length,
        n_kps,
        z_dim, 
        lag,
        nclass,
        hidden_dim=128,
        trans_prior='NP',
        infer_mode='R',
        bound=5,
        count_bins=8,
        order='linear',
        lr=1e-4,
        l1=1e-3,
        beta=0.0025,
        gamma=0.0075,
        sigma=1e-6,
        use_warm_start=False,
        spline_pth=None,
        kp_pth=None,
        decoder_dist='gaussian',
        correlation='Pearson'):
        '''Bi-directional inference network'''
        super().__init__()
        # Transition prior must be L (Linear) or NP (Nonparametric)
        assert trans_prior in ('L', 'NP')
        # Inference mode must be R (Recurrent)
        assert infer_mode in ('R')

        self.z_dim = z_dim
        self.nc = nc
        self.n_kps = n_kps
        self.input_dim = (nc, 64, 64)
        self.lr = lr
        self.lag = lag
        self.length = length
        self.nclass = nclass
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.l1 = l1
        self.correlation = correlation
        self.decoder_dist = decoder_dist
        self.infer_mode = infer_mode
        # Recurrent/Factorized inference
        if infer_mode == 'R':
            # Load pre-trained KP detector and freeze
            assert kp_pth is not None
            # self.kp = Keypointer(n_kps=n_kps, lim=[-1., 1., -1., 1.])
            self.kp = Keypointer.load_from_checkpoint(kp_pth, n_kps=n_kps)
            # for p in self.kp.parameters():
            #     p.requires_grad = False
            # Bi-directional hidden state rnn
            self.rnn = nn.ModuleList([nn.GRU(input_size=2, 
                                             hidden_size=hidden_dim, 
                                             num_layers=1, 
                                             batch_first=True, 
                                             bidirectional=True) for _ in range(n_kps)])
            # Inference net
            self.net = Inference(lag=lag,
                                 z_dim=n_kps*z_dim, 
                                 hidden_dim=n_kps*hidden_dim, 
                                 num_layers=2)

        # Initialize transition prior
        if trans_prior == 'L':
            self.transition_prior = MBDTransitionPrior(lags=lag, 
                                                       latent_size=z_dim*n_kps, 
                                                       bias=bias)
        elif trans_prior == 'NP':
            self.transition_prior = NPTransitionPrior(lags=lag, 
                                                      latent_size=z_dim*n_kps, 
                                                      num_layers=3, 
                                                      hidden_dim=64)
        
        # Spline flow model to learn the noise distribution
        self.spline_list = []
        for i in range(self.nclass):
            spline = ComponentWiseSpline(input_dim=z_dim*n_kps,
                                         bound=bound,
                                         count_bins=count_bins,
                                         order=order)

            if use_warm_start:
                spline.load_state_dict(torch.load(spline_pth, 
                                                  map_location=torch.device('cpu')))

                print("Load pretrained spline flow", flush=True)
            self.spline_list.append(spline)
        self.spline_list = nn.ModuleList(self.spline_list)

        # FactorVAE
        self.discriminator = Discriminator(z_dim = n_kps*z_dim*self.length)

        # Register skeleton logits
        self.logits = nn.Parameter(torch.randn(n_kps*self.z_dim, n_kps*self.z_dim*self.lag)+4)

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(self.z_dim*n_kps))
        self.register_buffer('base_dist_var', torch.eye(self.z_dim*n_kps))

    @property
    def base_dist(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def inference(self, ft, random_sampling=True):
        ## bidirectional lstm/gru 
        # input: ft: [batch_size, length, n_kps, 2]
        # output: (batch, seq_len, z_dim)
        output = [ ]
        for k in range(self.n_kps):
            out, _ = self.rnn[k](ft[:,:,k])
            output.append(out)
        output = torch.cat(output, dim=-1)
        batch_size, length, _ = output.shape
        # beta, hidden = self.gru(ft, hidden)
        ## sequential sampling & reparametrization
        ## transition: p(zt|z_tau)
        zs, mus, logvars = [], [], []
        for tau in range(self.lag):
            zs.append(torch.zeros((batch_size, self.z_dim*self.n_kps), device=output.device))

        for t in range(length):
            mid = torch.cat(zs[-self.lag:], dim=1)
            inputs = torch.cat([mid, output[:,t,:]], dim=1)    
            distributions = self.net(inputs)
            # Threshold mu inside [-1, 1] for heatmap limit
            mu = distributions[:, :self.n_kps*self.z_dim]
            # Skip connection for keypoint location
            mu[:,:self.n_kps*2] = ft[:,t].view(batch_size, -1) +  mu[:,:self.n_kps*2]
            logvar = distributions[:, self.n_kps*self.z_dim:]
            logvar = torch.clamp(logvar, min=-10)
            zt = self.reparameterize(mu, logvar, random_sampling)
            zs.append(zt)
            mus.append(mu)
            logvars.append(logvar)

        zs = torch.squeeze(torch.stack(zs, dim=1))
        # Strip the first L zero-initialized zt 
        zs = zs[:,self.lag:]
        mus = torch.squeeze(torch.stack(mus, dim=1))
        logvars = torch.squeeze(torch.stack(logvars, dim=1))
        return zs, mus, logvars
    
    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    def reconstruction_loss(self, x, x_recon, distribution):
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

    def forward(self, batch):
        x, y = batch['xt'], batch['yt']
        batch_size, length, nc, h, w = x.shape
        x_flat = x.view(-1, nc, h, w)
        if self.infer_mode == 'R':
            feat, kpts, hmap = self.kp.forward(x)
            zs, mus, logvars = self.inference(kpts)
        return zs, mus, logvars       

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y, ct = batch['s1']['xt'], batch['s1']['yt'], batch['s1']['ct']
        xr, yr, ctr = batch['s2']['xt'], batch['s2']['yt'], batch['s2']['ct']
        batch_size, length, nc, h, w = x.shape; ct = torch.squeeze(ct).to(torch.int64)
        x_flat = x.view(-1, nc, h, w)
        # Adjacent matrix
        adj_matrix = F.sigmoid(self.logits)
        # adj_matrix = F.gumbel_softmax(self.logits, tau=0.2, hard=True)[:,:,0]
        # Inference
        if self.infer_mode == 'R':
            feat, kpts, hmap = self.kp.forward(x)
            featr, kptsr, hmapr = self.kp.forward(xr)
            # zs: [B, length, n_kps*2]
            zs, mus, logvars = self.inference(kpts)
            kpts = zs[...,:self.n_kps*2].reshape(batch_size, length, self.n_kps, 2)
            x_recon = self.kp.reconstruct(kpts, feat, kptsr, featr)
         # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, nc, h, w)
        mus = mus.reshape(batch_size, length, self.n_kps*self.z_dim)
        logvars  = logvars.reshape(batch_size, length, self.n_kps*self.z_dim)
        zs = zs.reshape(batch_size, length, self.n_kps*self.z_dim)
        # x_gray = self.transforms(x_flat).view(batch_size, length, 1, h//4, w//4)
        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)
        # Past KLD
        p_dist = D.Normal(torch.zeros_like(mus[:,:self.lag]), torch.ones_like(logvars[:,:self.lag]))
        log_pz_normal = torch.sum(torch.sum(p_dist.log_prob(zs[:,:self.lag]),dim=-1),dim=-1)
        log_qz_normal = torch.sum(torch.sum(log_qz[:,:self.lag],dim=-1),dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()
        # Future KLD
        log_qz_laplace = log_qz[:,self.lag:]

        residuals, logabsdet = self.transition_prior(zs, adj_matrix)
        # residuals, logabsdet = self.transition_prior(zs)

        sum_log_abs_det_jacobians = 0
        one_hot = F.one_hot(ct, num_classes=self.nclass)
        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + logabsdet
        # Nonstationary branch
        es = [ ]
        logabsdet = [ ]
        for c in range(self.nclass):
            es_c, logabsdet_c = self.spline_list[c](residuals.contiguous().view(-1, self.z_dim*self.n_kps))
            es.append(es_c)
            logabsdet.append(logabsdet_c)

        es = torch.stack(es, axis=1)
        logabsdet = torch.stack(logabsdet, axis=1)
        mask = one_hot.unsqueeze(1).repeat(1,self.length,1).reshape(-1, self.nclass)
        es = (es * mask.unsqueeze(-1)).sum(1)
        logabsdet = (logabsdet * mask).sum(1)

        es = es.reshape(batch_size, length-self.lag, self.z_dim*self.n_kps)
        logabsdet = torch.sum(logabsdet.reshape(batch_size,length-self.lag), dim=1)

        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + logabsdet
        log_pz_laplace = torch.sum(self.base_dist.log_prob(es), dim=1) + sum_log_abs_det_jacobians
        kld_laplace = (torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace) / (length-self.lag)
        kld_laplace = kld_laplace.mean()

        l1_loss = torch.norm(adj_matrix, 1)
        # VAE training
        if optimizer_idx == 0:
            for p in self.discriminator.parameters():
                p.requires_grad = False
            D_z = self.discriminator(residuals.contiguous().view(batch_size, -1))
            tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()
            loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace + self.sigma * tc_loss + self.l1*l1_loss
            # loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace + self.sigma * tc_loss
            self.log("train_elbo_loss", loss)
            self.log("train_recon_loss", recon_loss)
            self.log("train_kld_normal", kld_normal)
            self.log("train_kld_laplace", kld_laplace)
            self.log("train_l1_loss", l1_loss)
            self.log("v_tc_loss", tc_loss)
            return loss

        # Discriminator training
        if optimizer_idx == 1:
            for p in self.discriminator.parameters():
                p.requires_grad = True

            residuals = residuals.detach()
            D_z = self.discriminator(residuals.contiguous().view(batch_size, -1))
            # Permute the other data batch
            ones = torch.ones(batch_size, dtype=torch.long).to(batch['s2']['yt'].device)
            zeros = torch.zeros(batch_size, dtype=torch.long).to(batch['s2']['yt'].device)
            zs_perm, _, _ = self.forward(batch['s2'])
            zs_perm = zs_perm.reshape(batch_size, length, self.n_kps*self.z_dim)
            residuals_perm, _ = self.transition_prior(zs_perm, adj_matrix)
            # residuals_perm, _ = self.transition_prior(zs_perm)
            residuals_perm = permute_dims(residuals_perm.contiguous().view(batch_size, -1)).detach()
            D_z_pperm = self.discriminator(residuals_perm)
            D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))            
            self.log("d_tc_loss", D_tc_loss)
            return D_tc_loss
    
    def validation_step(self, batch, batch_idx):
        x, y, ct = batch['s1']['xt'], batch['s1']['yt'], batch['s1']['ct']
        xr, yr, ctr = batch['s2']['xt'], batch['s2']['yt'], batch['s2']['ct']
        batch_size, length, nc, h, w = x.shape; ct = torch.squeeze(ct).to(torch.int64)
        x_flat = x.view(-1, nc, h, w)
        # Adjacent matrix
        adj_matrix = F.sigmoid(self.logits)
        # adj_matrix = F.gumbel_softmax(self.logits, tau=0.2, hard=True)[:,:,0]
        # Inference
        if self.infer_mode == 'R':
            feat, kpts, hmap = self.kp.forward(x)
            featr, kptsr, hmapr = self.kp.forward(xr)
            # zs: [B, length, n_kps*2]
            zs, mus, logvars = self.inference(kpts)
            kpts = zs[...,:self.n_kps*2].reshape(batch_size, length, self.n_kps, 2)
            x_recon = self.kp.reconstruct(kpts, feat, kptsr, featr)
         # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, nc, h, w)
        mus = mus.reshape(batch_size, length, self.n_kps*self.z_dim)
        logvars  = logvars.reshape(batch_size, length, self.n_kps*self.z_dim)
        zs = zs.reshape(batch_size, length, self.n_kps*self.z_dim)
        # x_gray = self.transforms(x_flat).view(batch_size, length, 1, h//4, w//4)
        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)
        # Past KLD
        p_dist = D.Normal(torch.zeros_like(mus[:,:self.lag]), torch.ones_like(logvars[:,:self.lag]))
        log_pz_normal = torch.sum(torch.sum(p_dist.log_prob(zs[:,:self.lag]),dim=-1),dim=-1)
        log_qz_normal = torch.sum(torch.sum(log_qz[:,:self.lag],dim=-1),dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()
        # Future KLD
        log_qz_laplace = log_qz[:,self.lag:]

        residuals, logabsdet = self.transition_prior(zs, adj_matrix)
        # residuals, logabsdet = self.transition_prior(zs)

        sum_log_abs_det_jacobians = 0
        one_hot = F.one_hot(ct, num_classes=self.nclass)
        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + logabsdet
        # Nonstationary branch
        es = [ ]
        logabsdet = [ ]
        for c in range(self.nclass):
            es_c, logabsdet_c = self.spline_list[c](residuals.contiguous().view(-1, self.z_dim*self.n_kps))
            es.append(es_c)
            logabsdet.append(logabsdet_c)

        es = torch.stack(es, axis=1)
        logabsdet = torch.stack(logabsdet, axis=1)
        mask = one_hot.unsqueeze(1).repeat(1,self.length,1).reshape(-1, self.nclass)
        es = (es * mask.unsqueeze(-1)).sum(1)
        logabsdet = (logabsdet * mask).sum(1)

        es = es.reshape(batch_size, length-self.lag, self.z_dim*self.n_kps)
        logabsdet = torch.sum(logabsdet.reshape(batch_size,length-self.lag), dim=1)

        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + logabsdet
        log_pz_laplace = torch.sum(self.base_dist.log_prob(es), dim=1) + sum_log_abs_det_jacobians
        kld_laplace = (torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace) / (length-self.lag)
        kld_laplace = kld_laplace.mean()
        l1_loss = torch.norm(adj_matrix, 1)
        # VAE training
        loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace + self.l1*l1_loss
        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mus.view(-1, self.n_kps*self.z_dim).T.detach().cpu().numpy()
        # zt_true = batch['s1']["yt"].reshape(batch_size,length,-1).view(-1, self.n_kps*self.z_dim).T.detach().cpu().numpy()
        zt_true = batch['s1']["yt"][...,:2].reshape(batch_size,length,-1).view(-1, self.n_kps*2).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)

        self.log("val_mcc", mcc) 
        self.log("val_elbo_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kld_normal", kld_normal)
        self.log("val_kld_laplace", kld_laplace)

        return loss
    
    def sample(self, n=64):
        with torch.no_grad():
            e = torch.randn(n, self.z_dim, device=self.device)
            eps, _ = self.spline.inverse(e)
        return eps

    def reconstruct(self, batch):
        zs, mus, logvars = self.forward(batch)
        zs_flat = zs.contiguous().view(-1, self.z_dim)
        x_recon = self.dec(zs_flat)
        x_recon = x_recon.view(batch_size, self.length, self.input_dim)       
        return x_recon

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        opt_d = torch.optim.SGD(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=self.lr/2)
        return [opt_v, opt_d], []