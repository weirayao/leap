"""Temporal VAE with gaussian margial and laplacian transition prior"""

import torch
import numpy as np
import ipdb as pdb
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import pytorch_lightning as pl
import torch.distributions as D
from torch.nn import functional as F
from .components.transition import PNLTransitionPrior, MBDTransitionPrior
from .components.mlp import MLPEncoder, MLPDecoder, Inference
from .components.tc import Discriminator, permute_dims
from .components.beta import BetaVAE_MLP
from .metrics.correlation import compute_mcc
from .components.transforms import ComponentWiseSpline
from .components.mlp import MLPEncoder, MLPDecoder, NLayerLeakyMLP, NLayerLeakyMLP
from .components.transition import LinearTransitionPrior, PNLTransitionPrior, INTransitionPrior


class SRNNSyntheticNS(pl.LightningModule):

    def __init__(
        self, 
        input_dim,
        length,
        z_dim, 
        lag,
        nclass,
        hidden_dim=128,
        trans_prior='L',
        infer_mode='R',
        bound=5,
        count_bins=8,
        order='linear',
        lr=1e-4,
        beta=0.0025,
        gamma=0.0075,
        sigma=1e-6,
        bias=False,
        use_warm_start=False,
        spline_pth=None,
        decoder_dist='gaussian',
        correlation='Pearson'):
        '''Bi-directional inference network'''
        super().__init__()
        # Transition prior must be L (Linear), PNL (Post-nonlinear) or IN (Interaction)
        assert trans_prior in ('L', 'PNL','IN')
        # Inference mode must be R (Recurrent) or F (Factorized)
        assert infer_mode in ('R', 'F')

        self.z_dim = z_dim
        self.lag = lag
        self.input_dim = input_dim
        self.lr = lr
        self.lag = lag
        self.length = length
        self.nclass = nclass
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.correlation = correlation
        self.decoder_dist = decoder_dist
        self.infer_mode = infer_mode

        # Recurrent/Factorized inference
        if infer_mode == 'R':
            self.enc = MLPEncoder(latent_size=z_dim, 
                                num_layers=3, 
                                hidden_dim=hidden_dim)

            self.dec = MLPDecoder(latent_size=z_dim, 
                                num_layers=2,
                                hidden_dim=hidden_dim)

            # Bi-directional hidden state rnn
            self.rnn = nn.GRU(input_size=z_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=True)
            
            # Inference net
            self.net = Inference(lag=lag,
                                 z_dim=z_dim, 
                                 hidden_dim=hidden_dim, 
                                 num_layers=1)

        elif infer_mode == 'F':
            self.net = BetaVAE_MLP(input_dim=input_dim, 
                                   z_dim=z_dim, 
                                   hidden_dim=hidden_dim)

        # Initialize transition prior
        if trans_prior == 'L':
            self.transition_prior = MBDTransitionPrior(lags=lag, 
                                                       latent_size=z_dim, 
                                                       bias=bias)
        elif trans_prior == 'PNL':
            self.transition_prior = PNLTransitionPrior(lags=lag, 
                                                       latent_size=z_dim, 
                                                       num_layers=1, 
                                                       hidden_dim=hidden_dim)
        elif trans_prior == 'IN':
            self.transition_prior = INTransitionPrior()
        
        # Spline flow model to learn the noise distribution
        self.spline_list = []
        for i in range(self.nclass):
            self.spline = ComponentWiseSpline(input_dim=z_dim,
                                            bound=bound,
                                            count_bins=count_bins,
                                            order=order)

            if use_warm_start:
                self.spline.load_state_dict(torch.load(spline_pth, 
                                            map_location=torch.device('cpu')))

                print("Load pretrained spline flow", flush=True)
            self.spline_list.append(self.spline)

        # FactorVAE
        self.discriminator = Discriminator(z_dim = z_dim*self.length)

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
            zs.append(torch.ones((batch_size, self.z_dim), device=output.device))

        for t in range(length):
            mid = torch.cat(zs[-self.lag:], dim=1)
            inputs = torch.cat([mid, output[:,t,:]], dim=1)    
            distributions = self.net(inputs)
            mu = distributions[:, :self.z_dim]
            logvar = distributions[:, self.z_dim:]
            zt = self.reparameterize(mu, logvar, random_sampling)
            zs.append(zt)
            mus.append(mu)
            logvars.append(logvar)

        zs = torch.squeeze(torch.stack(zs, dim=1))
        # Strip the first L zero-initialized zt 
        zs = zs[:,self.lag:]
        mus = torch.squeeze(torch.stack(mus, dim=1))
        logvars = torch.squeeze(torch.stack(logvars, dim=1))
        # logvars = torch.nn.functional.softplus(logvars)
        return zs, mus, logvars
    
    def reparameterize(self, mu, logvar, random_sampling=True):
        if random_sampling:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mu + eps*std
            return z
        else:
            return mu

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
        x, y, c = batch['xt'], batch['yt'], batch['ct']
        batch_size, length, _, nclass = x.shape
        x_recon_list = []; zs_list = []
        mus_list = []; logvars_list = []
        for i in range(nclass):
            x_flat = x[...,i].view(-1, self.input_dim)
            # Inference
            if self.infer_mode == 'R':
                ft = self.enc(x_flat)
                ft = ft.view(batch_size, length, -1)
                zs, mus, logvars = self.inference(ft, random_sampling=True)
            elif self.infer_mode == 'F':
                _, mus, logvars, zs = self.net(x_flat)
            
            mus_list.append(mus); logvars_list.append(logvars); zs_list.append(zs); 
        
        mus = torch.cat(mus_list, dim=0)
        logvars = torch.cat(logvars_list, dim=0)
        zs = torch.cat(zs_list, dim=0)

        # Reshape to time-series format
        mus = mus.reshape(batch_size, length, self.z_dim, nclass)
        logvars  = logvars.reshape(batch_size, length, self.z_dim, nclass)
        zs = zs.reshape(batch_size, length, self.z_dim, nclass)
        
        return zs, mus, logvars       

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y, c = batch['s1']['xt'], batch['s1']['yt'], batch['s1']['ct']
        batch_size, length, _, nclass = x.shape

        x_recon_list = []; zs_list = []
        mus_list = []; logvars_list = []

        for i in range(nclass):
            x_flat = x[...,i].view(-1, self.input_dim)
            # Inference
            if self.infer_mode == 'R':
                ft = self.enc(x_flat)
                ft = ft.view(batch_size, length, -1)
                zs, mus, logvars = self.inference(ft)
                zs_flat = zs.contiguous().view(-1, self.z_dim)
                x_recon = self.dec(zs_flat)
            elif self.infer_mode == 'F':
                x_recon, mus, logvars, zs = self.net(x_flat)
            
            x_recon_list.append(x_recon); mus_list.append(mus); 
            logvars_list.append(logvars); zs_list.append(zs); 
        
        x_recon = torch.cat(x_recon_list, dim=0); mus = torch.cat(mus_list, dim=0)
        logvars = torch.cat(logvars_list, dim=0); zs = torch.cat(zs_list, dim=0)

        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, self.input_dim, nclass)
        mus = mus.reshape(batch_size, length, self.z_dim, nclass)
        logvars  = logvars.reshape(batch_size, length, self.z_dim, nclass)
        zs = zs.reshape(batch_size, length, self.z_dim, nclass)

        # VAE ELBO loss: recon_loss + kld_loss
        demo_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist)
        recon_sum = torch.zeros(demo_loss.shape).to(x.device)
        for i in range(nclass):
            recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
            (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
            recon_sum += recon_loss
            break
        
        q_dist = D.Normal(mus, torch.abs(torch.exp(logvars / 2)))
        log_qz = q_dist.log_prob(zs)
        # except:
        #     pdb.set_trace()
        # Past KLD
        kld_normal_sum = torch.zeros(demo_loss.shape).to(x.device)
        for i in range(nclass):
            p_dist = D.Normal(torch.zeros_like(mus[:,:self.lag,:,i]), torch.ones_like(logvars[:,:self.lag,:,i]))
            log_pz_normal = torch.sum(torch.sum(p_dist.log_prob(zs[:,:self.lag,:,i]),dim=-1),dim=-1)
            log_qz_normal = torch.sum(torch.sum(log_qz[:,:self.lag,:,i],dim=-1),dim=-1)
            kld_normal = log_qz_normal - log_pz_normal
            kld_normal = kld_normal.mean()
            kld_normal_sum += kld_normal
            break
        # Future KLD
        residuals_list = []
        kld_lap_sum = torch.zeros(demo_loss.shape).to(x.device)
        for i in range(nclass):
            sum_log_abs_det_jacobians = 0
            log_qz_laplace = log_qz[:, self.lag:, :, i]
            residuals, logabsdet = self.transition_prior(zs[...,i])
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + logabsdet
            
            es, logabsdet = self.spline_list[i].to(x.device)(residuals.contiguous().view(-1, self.z_dim))
            es = es.reshape(batch_size, length-self.lag, self.z_dim)
            logabsdet = torch.sum(logabsdet.reshape(batch_size,length-self.lag), dim=1)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + logabsdet
            log_pz_laplace = torch.sum(self.base_dist.log_prob(es), dim=1) + sum_log_abs_det_jacobians
            kld_laplace = (torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace) / (length-self.lag)
            kld_laplace = kld_laplace.mean()
            kld_lap_sum += kld_laplace
            residuals_list.append(residuals)
            break
    
        # VAE training
        if optimizer_idx == 0:
            for p in self.discriminator.parameters():
                p.requires_grad = False
            tc_loss_sum = torch.zeros(demo_loss.shape).to(x.device)
            for i in range(nclass):
                D_z = self.discriminator(residuals_list[i].contiguous().view(batch_size, -1))
                tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()
                tc_loss_sum += tc_loss
                break

            loss = (recon_sum + self.beta * kld_normal + self.gamma * kld_lap_sum + self.sigma * tc_loss_sum)/nclass
            
            self.log("train_elbo_loss", loss)
            self.log("train_recon_loss", recon_sum)
            self.log("train_kld_normal", kld_normal)
            self.log("train_kld_laplace", kld_lap_sum)
            self.log("v_tc_loss", tc_loss_sum)
            return loss

        # Discriminator training
        if optimizer_idx == 1:
            for p in self.discriminator.parameters():
                p.requires_grad = True
            D_tc_loss_sum = torch.zeros(recon_loss.shape).to(x.device)
            for i in range(nclass):
                residuals = residuals_list[i].detach()
                D_z = self.discriminator(residuals.contiguous().view(batch_size, -1))
                # Permute the other data batch
                ones = torch.ones(batch_size, dtype=torch.long).to(batch['s2']['yt'].device)
                zeros = torch.zeros(batch_size, dtype=torch.long).to(batch['s2']['yt'].device)
                zs_perm, _, _ = self.forward(batch['s2'])
                zs_perm = zs_perm[...,i]
                residuals_perm, _ = self.transition_prior(zs_perm)
                residuals_perm = permute_dims(residuals_perm.contiguous().view(batch_size, -1)).detach()
                D_z_pperm = self.discriminator(residuals_perm)
                D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))            
                D_tc_loss_sum += D_tc_loss
                break
            
            self.log("d_tc_loss", D_tc_loss_sum)

            return D_tc_loss_sum
    
    def validation_step(self, batch, batch_idx):
        x, y, ct = batch['s1']['xt'], batch['s1']['yt'], batch['s1']['ct']
        batch_size, length, _, nclass = x.shape

        x_recon_list = []; zs_list = []
        mus_list = []; logvars_list = []

        for i in range(nclass):
            x_flat = x[...,i].view(-1, self.input_dim)
            # Inference
            if self.infer_mode == 'R':
                ft = self.enc(x_flat)
                ft = ft.view(batch_size, length, -1)
                zs, mus, logvars = self.inference(ft)
                zs_flat = zs.contiguous().view(-1, self.z_dim)
                x_recon = self.dec(zs_flat)
            elif self.infer_mode == 'F':
                x_recon, mus, logvars, zs = self.net(x_flat)
            
            x_recon_list.append(x_recon); mus_list.append(mus); 
            logvars_list.append(logvars); zs_list.append(zs); 
        
        x_recon = torch.cat(x_recon_list, dim=0); mus = torch.cat(mus_list, dim=0)
        logvars = torch.cat(logvars_list, dim=0); zs = torch.cat(zs_list, dim=0)

        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, self.input_dim, nclass)
        mus = mus.reshape(batch_size, length, self.z_dim, nclass)
        logvars  = logvars.reshape(batch_size, length, self.z_dim, nclass)
        zs = zs.reshape(batch_size, length, self.z_dim, nclass)

        # VAE ELBO loss: recon_loss + kld_loss
        demo_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist)
        recon_sum = torch.zeros(demo_loss.shape).to(x.device)
        for i in range(nclass):
            recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
            (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
            recon_sum += recon_loss
            break

        q_dist = D.Normal(mus, torch.abs(torch.exp(logvars / 2)))
        log_qz = q_dist.log_prob(zs)

        # Past KLD
        kld_normal_sum = torch.zeros(demo_loss.shape).to(x.device)
        for i in range(nclass):
            p_dist = D.Normal(torch.zeros_like(mus[:,:self.lag,:,i]), torch.ones_like(logvars[:,:self.lag,:,i]))
            log_pz_normal = torch.sum(torch.sum(p_dist.log_prob(zs[:,:self.lag,:,i]),dim=-1),dim=-1)
            log_qz_normal = torch.sum(torch.sum(log_qz[:,:self.lag,:,i],dim=-1),dim=-1)
            kld_normal = log_qz_normal - log_pz_normal
            kld_normal = kld_normal.mean()
            kld_normal_sum += kld_normal
            break
        # Future KLD
        kld_lap_sum = torch.zeros(demo_loss.shape).to(x.device)
        for i in range(nclass):
            sum_log_abs_det_jacobians = 0
            log_qz_laplace = log_qz[:, self.lag:, :, i]
            residuals, logabsdet = self.transition_prior(zs[...,i])
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + logabsdet
            # pdb.set_trace()
            es, logabsdet = self.spline_list[i].to(x.device)(residuals.contiguous().view(-1, self.z_dim))
            es = es.reshape(batch_size, length-self.lag, self.z_dim)
            logabsdet = torch.sum(logabsdet.reshape(batch_size,length-self.lag), dim=1)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + logabsdet
            log_pz_laplace = torch.sum(self.base_dist.log_prob(es), dim=1) + sum_log_abs_det_jacobians
            kld_laplace = (torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace) / (length-self.lag)
            kld_laplace = kld_laplace.mean()
            kld_lap_sum += kld_laplace
            break
        
        loss = (recon_sum + self.beta * kld_normal_sum + self.gamma * kld_lap_sum)/nclass

        mcc_sum = 0
        for i in range(nclass):
            zt_recon = mus[...,i].reshape(-1, self.z_dim).T.detach().cpu().numpy()
            zt_true = batch['s1']["yt"][...,i].view(-1, self.z_dim).T.detach().cpu().numpy()
            mcc = compute_mcc(zt_recon, zt_true, self.correlation)
            mcc_sum += mcc
            break

        self.log("val_mcc", mcc_sum/nclass) 
        self.log("val_elbo_loss", loss)
        self.log("val_recon_loss", recon_sum)
        self.log("val_kld_normal", kld_normal_sum)
        self.log("val_kld_laplace", kld_lap_sum)

        return loss
    
    def sample(self, n=64):
        with torch.no_grad():
            e = torch.randn(n, self.z_dim, device=self.device)
            eps, _ = self.spline.inverse(e)
        return eps

    # def reconstruct(self, batch):
    #     x, y = batch['xt'], batch['yt']
    #     batch_size, length, _ = x.shape
    #     zs, mus, logvars = self.forward(batch)
    #     zs_flat = zs.contiguous().view(-1, self.z_dim)
    #     x_recon = self.dec(zs_flat)
    #     x_recon = x_recon.view(batch_size, length, self.input_dim)       
    #     return x_recon

    def configure_optimizers(self):
        opt_v = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999))
        opt_d = torch.optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=self.lr/2, betas=(0.9, 0.999))
        # opt_d = torch.optim.SGD(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=self.lr/2)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [opt_v, opt_d], []
