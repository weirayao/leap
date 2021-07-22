"""Temporal VAE with gaussian margial and laplacian transition prior"""

import torch
import numpy as np
import ipdb as pdb
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributions as D
from torch.nn import functional as F
from .components.transition import LinearTransitionPrior, PNLTransitionPrior
from .components.mlp import MLPEncoder, MLPDecoder, NLayerLeakyMLP
from .metrics.correlation import compute_mcc
from .components.transforms import ComponentWiseSpline

import ipdb as pdb

class SRNNSynthetic(pl.LightningModule):

    def __init__(
        self, 
        input_dim,
        length,
        z_dim, 
        lag,
        hidden_dim=128,
        trans_prior='L',
        bound=5,
        count_bins=8,
        order='linear',
        lr=1e-4,
        beta=0.1,
        bias=False,
        use_warm_start=False,
        spline_pth=None,
        decoder_dist='gaussian',
        correlation='Pearson'):
        '''Bi-directional inference network'''
        super().__init__()
        # Transition prior must be L (Linear), PNL (Post-nonlinear) or IN (Interaction)
        assert trans_prior in ('L', 'PNL','IN')

        self.z_dim = z_dim
        self.lag = lag
        self.input_dim = input_dim
        self.lr = lr
        self.lag = lag
        self.length = length
        self.beta = beta
        self.correlation = correlation
        self.decoder_dist = decoder_dist
        self.enc = MLPEncoder(input_dim=input_dim,
                              latent_size=z_dim, 
                              num_layers=4, 
                              hidden_dim=hidden_dim)

        self.dec = MLPDecoder(latent_size=z_dim, 
                              num_layers=2)
        
        # Bi-directional hidden state rnn
        self.rnn = nn.GRU(input_size=z_dim, 
                          hidden_size=hidden_dim, 
                          num_layers=1, 
                          batch_first=True, 
                          bidirectional=True)
        
        # Sequential inference net 
        self.q_mu = NLayerLeakyMLP(in_features=lag*z_dim+2*hidden_dim, 
                                   out_features=z_dim, 
                                   num_layers=3, 
                                   hidden_dim=hidden_dim)

        self.q_logvar = NLayerLeakyMLP(in_features=lag*z_dim+2*hidden_dim, 
                                       out_features=z_dim, 
                                       num_layers=3, 
                                       hidden_dim=hidden_dim)

        # Initialize transition prior
        if trans_prior == 'L':
            self.transition_prior = LinearTransitionPrior(lags=lag, 
                                                          latent_size=z_dim, 
                                                          bias=bias)
        elif trans_prior == 'PNL':
            self.transition_prior = PNLTransitionPrior(lags=lag, 
                                                       latent_size=z_dim, 
                                                       num_layers=3, 
                                                       hidden_dim=hidden_dim)
        elif trans_prior == 'IN':
            self.transition_prior = INTransitionPrior()
        
        self.spline = ComponentWiseSpline(input_dim=z_dim,
                                          bound=bound,
                                          count_bins=count_bins,
                                          order=order)

        if use_warm_start:
            self.spline.load_state_dict(torch.load(spline_pth, 
                                        map_location=torch.device('cpu')))

            print("Load pretrained spline flow", flush=True)

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(self.z_dim))
        self.register_buffer('base_dist_var', torch.eye(self.z_dim))

    @property
    def base_dist(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def inference(self, ft, random_sampling=True):
        ## bidirectional lstm/gru 
        # input: (batch, seq_len, z_dim)
        # output: (batch, seq_len, z_dim)
        output, h_n = self.rnn(ft)
        batch_size, length, _ = output.shape
        # beta, hidden = self.gru(ft, hidden)
        ## sequential sampling & reparametrization
        ## transition: p(zt|z_tau)
        zs, mus, logvars = [], [], []
        for tau in range(self.lag):
            zs.append(torch.zeros((batch_size, self.z_dim), device=output.device))

        for t in range(length):
            mid = torch.cat(zs[-self.lag:], dim=1)
            inputs = torch.cat([mid, output[:,t,:]], dim=1)       
            mu = self.q_mu(inputs)
            logvar = self.q_logvar(inputs)
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
        batch_size, length, _ = x.shape
        x = x.view(-1, self.input_dim)
        ft = self.enc(x)
        ft = ft.view(batch_size, length, -1)
        zs, mus, logvars = self.inference(ft, random_sampling=False)
        return zs, mus, logvars       

    def training_step(self, batch, batch_idx):
        x, y = batch['xt'], batch['yt']
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        ft = self.enc(x_flat)
        ft = ft.view(batch_size, length, -1)
        zs, mus, logvars = self.inference(ft)
        zs_flat = zs.contiguous().view(-1, self.z_dim)
        x_recon = self.dec(zs_flat)
        x_recon = x_recon.view(batch_size, length, self.input_dim)
        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = self.reconstruction_loss(x, x_recon, self.decoder_dist)
        residuals = self.transition_prior(zs)
        es, logabsdet = self.spline(residuals.contiguous().view(-1, self.z_dim))
        es = es.reshape(batch_size, length, self.z_dim)
        logabsdet = torch.sum(logabsdet.reshape(batch_size,length), dim=1)
        log_pz = torch.sum(self.base_dist.log_prob(es), dim=1) + logabsdet
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)
        kld_loss = torch.sum(torch.sum(log_qz,dim=-1),dim=-1) - log_pz
        kld_loss = kld_loss[~torch.isnan(kld_loss)].mean()
        
        loss = self.length * recon_loss + self.beta * kld_loss

        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_loss", kld_loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['xt'], batch['yt']
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        ft = self.enc(x_flat)
        ft = ft.view(batch_size, length, -1)
        zs, mus, logvars = self.inference(ft)
        zs_flat = zs.contiguous().view(-1, self.z_dim)
        x_recon = self.dec(zs_flat)
        x_recon = x_recon.view(batch_size, length, self.input_dim)
        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = self.reconstruction_loss(x, x_recon, self.decoder_dist)
        residuals = self.transition_prior(zs)
        es, logabsdet = self.spline(residuals.contiguous().view(-1, self.z_dim))
        es = es.reshape(batch_size, length, self.z_dim)
        logabsdet = torch.sum(logabsdet.reshape(batch_size,length), dim=1)
        log_pz = torch.sum(self.base_dist.log_prob(es), dim=1) + logabsdet
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)
        kld_loss = torch.sum(torch.sum(log_qz,dim=-1),dim=-1) - log_pz
        kld_loss = kld_loss[~torch.isnan(kld_loss)].mean()
        
        loss = self.length * recon_loss + self.beta * kld_loss

        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()
        zt_true = batch["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)

        self.log("val_mcc", mcc) 
        self.log("val_elbo_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kld_loss", kld_loss)

        return loss
    
    def sample(self, n=64):
        with torch.no_grad():
            e = torch.randn(n, self.z_dim, device=self.device)
            eps, _ = self.spline.inverse(e)
        return eps

    def reconstruct(self):
        zs, mus, logvars = self.forward(batch)
        zs_flat = zs.contiguous().view(-1, self.z_dim)
        x_recon = self.dec(zs_flat)
        x_recon = x_recon.view(batch_size, length, self.input_dim)       
        return x_recon

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        return optimizer