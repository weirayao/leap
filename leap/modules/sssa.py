import torch
import numpy as np
import ipdb as pdb
import torch.nn as nn
import torch.nn.init as init
import pytorch_lightning as pl
import torch.distributions as D
from torch.nn import functional as F
from .components.beta import BetaVAE_MLP
from .metrics.correlation import compute_mcc
from .metrics.block import compute_r2
from .components.tc import Discriminator, permute_dims
from .components.transforms import ComponentWiseSpline
from .components.hsic import RbfHSIC

class SSA(pl.LightningModule):

    def __init__(
        self, 
        input_dim,
        c_dim,
        s_dim, 
        nclass,
        hidden_dim=128,
        bound=5,
        count_bins=8,
        order='linear',
        lr=1e-4,
        beta=0.0025,
        gamma=0.001,
        sigma=1e-6,
        sigma_x=0.1,
        sigma_y=None,
        use_warm_start=False,
        spline_pth=None,
        decoder_dist='gaussian',
        correlation='Pearson'):
        '''Stationary subspace analysis'''
        super().__init__()
        self.c_dim = c_dim
        self.s_dim = s_dim
        self.z_dim = c_dim + s_dim
        self.input_dim = input_dim
        self.lr = lr
        self.nclass = nclass
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.correlation = correlation
        self.decoder_dist = decoder_dist
        self.fc = nn.Linear(s_dim, nclass)
        self.ce = nn.CrossEntropyLoss()
        # Inference
        self.net = BetaVAE_MLP(input_dim=input_dim, 
                               z_dim=self.z_dim, 
                               hidden_dim=hidden_dim)
        
        # Spline flow model to learn the noise distribution
        self.spline_list = []
        for i in range(self.nclass):
            spline = ComponentWiseSpline(input_dim=s_dim,
                                         bound=bound,
                                         count_bins=count_bins,
                                         order=order)

            if use_warm_start:
                spline.load_state_dict(torch.load(spline_pth, 
                                                  map_location=torch.device('cpu')))

                print("Load pretrained spline flow", flush=True)
            self.spline_list.append(spline)
        self.spline_list = nn.ModuleList(self.spline_list)
        # HSIC to enforce subspace independence
        self.hsic = RbfHSIC(sigma_x=sigma_x, sigma_y=None)
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(self.s_dim))
        self.register_buffer('base_dist_var', torch.eye(self.s_dim))

    @property
    def base_dist(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)
    
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
        x, y = batch['x'], batch['y']
        batch_size, _ = x.shape
        _, mus, logvars, zs = self.net(x)
        return zs, mus, logvars       

    def training_step(self, batch, batch_idx):
        x, y, c = batch['x'], batch['y'], batch['c']
        batch_size, _ = x.shape
        c = torch.squeeze(c).to(torch.int64)
        x_recon, mus, logvars, zs = self.net(x)
        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = self.reconstruction_loss(x, x_recon, self.decoder_dist)
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)
        # Content KLD
        p_dist = D.Normal(torch.zeros_like(mus[:,:self.c_dim]), torch.ones_like(logvars[:,:self.c_dim]))
        log_pz_content = torch.sum(p_dist.log_prob(zs[:,:self.c_dim]),dim=-1)
        log_qz_content = torch.sum(log_qz[:,:self.c_dim],dim=-1)
        kld_content = log_qz_content - log_pz_content
        kld_content = kld_content.mean()
        # Style KLD
        log_qz_style = log_qz[:,self.c_dim:]
        residuals = zs[:,self.c_dim:]
        sum_log_abs_det_jacobians = 0
        one_hot = F.one_hot(c, num_classes=self.nclass)
        # Nonstationary branch
        es = [ ]
        logabsdet = [ ]
        for c_id in range(self.nclass):
            es_c, logabsdet_c = self.spline_list[c_id](residuals)
            es.append(es_c)
            logabsdet.append(logabsdet_c)
        es = torch.stack(es, axis=1)
        logabsdet = torch.stack(logabsdet, axis=1)
        mask = one_hot.reshape(-1, self.nclass)
        es = (es * mask.unsqueeze(-1)).sum(1)
        logabsdet = (logabsdet * mask).sum(1)
        es = es.reshape(batch_size, self.s_dim)
        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + logabsdet
        log_pz_style = self.base_dist.log_prob(es) + sum_log_abs_det_jacobians
        kld_style = torch.sum(log_qz_style, dim=-1) - log_pz_style
        kld_style = kld_style.mean()
        # Classification branch
        ce_loss = self.ce(self.fc(residuals), c)
        # VAE training
        loss = recon_loss + self.beta * kld_content + self.gamma * kld_style + self.sigma * ce_loss
        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_content", kld_content)
        self.log("train_kld_style", kld_style)
        self.log("train_ce", ce_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, c = batch['x'], batch['y'], batch['c']
        batch_size, _ = x.shape
        c = torch.squeeze(c).to(torch.int64)
        x_recon, mus, logvars, zs = self.net(x)
        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = self.reconstruction_loss(x, x_recon, self.decoder_dist)
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)
        # Content KLD
        p_dist = D.Normal(torch.zeros_like(mus[:,:self.c_dim]), torch.ones_like(logvars[:,:self.c_dim]))
        log_pz_content = torch.sum(p_dist.log_prob(zs[:,:self.c_dim]),dim=-1)
        log_qz_content = torch.sum(log_qz[:,:self.c_dim],dim=-1)
        kld_content = log_qz_content - log_pz_content
        kld_content = kld_content.mean()
        # Style KLD
        log_qz_style = log_qz[:,self.c_dim:]
        residuals = zs[:,self.c_dim:]
        sum_log_abs_det_jacobians = 0
        one_hot = F.one_hot(c, num_classes=self.nclass)
        # Nonstationary branch
        es = [ ]
        logabsdet = [ ]
        for c_id in range(self.nclass):
            es_c, logabsdet_c = self.spline_list[c_id](residuals)
            es.append(es_c)
            logabsdet.append(logabsdet_c)
        es = torch.stack(es, axis=1)
        logabsdet = torch.stack(logabsdet, axis=1)
        mask = one_hot.reshape(-1, self.nclass)
        es = (es * mask.unsqueeze(-1)).sum(1)
        logabsdet = (logabsdet * mask).sum(1)
        es = es.reshape(batch_size, self.s_dim)
        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + logabsdet
        log_pz_style = self.base_dist.log_prob(es) + sum_log_abs_det_jacobians
        kld_style = torch.sum(log_qz_style, dim=-1) - log_pz_style
        kld_style = kld_style.mean()
        # Classification branch
        ce_loss = self.ce(self.fc(residuals), c)
        # VAE training
        loss = recon_loss + self.beta * kld_content + self.gamma * kld_style + self.sigma * ce_loss
        # Compute Kernel Regression R^2
        r2 = compute_r2(mus[:,:self.c_dim], batch["y"][:,:self.c_dim])
        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mus[:,self.c_dim:].T.detach().cpu().numpy()
        zt_true = batch["y"][:,self.c_dim:].T.detach().cpu().numpy()
        # pdb.set_trace()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)

        self.log("val_mcc", mcc)
        self.log("val_r2", r2)  
        self.log("val_elbo_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kld_content", kld_content)
        self.log("val_kld_style", kld_style)
        self.log("val_ce", ce_loss)

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
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [opt_v], []