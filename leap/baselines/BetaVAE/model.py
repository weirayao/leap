"""
BetaVAE.py
- https://github.com/1Konny/Beta-VAE
- No Transition, Contrastive Learning and Condition
"""
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

"utils file (SAME)"
from ..metrics.correlation import compute_mcc
"BetaVAE list"
from .net import BetaVAEMLP, BetaVAE_CNN, BetaVAE_KP

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(
            x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class BetaVAE(pl.LightningModule):
    def __init__(self, 
                 input_dim, 
                 z_dim, 
                 hidden_dim, 
                 beta,
                 beta1,
                 beta2,
                 lr,
                 correlation):
        # Networks & Optimizers
        super(BetaVAE, self).__init__()
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.correlation = correlation
        self.decoder_dist = 'gaussian'
        self.lr = lr
        self.net = BetaVAEMLP(self.input_dim, self.z_dim, self.hidden_dim)
    
    def training_step(self, batch, batch_idx):
        x = batch['s1']['xt'].reshape(-1, self.input_dim)
        x_recon, mu, logvar = self.net(x)
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
        total_kld, dimension_wise_kld, mean_kld = kl_divergence(mu, logvar)
        vae_loss = recon_loss + self.beta * total_kld
                
        self.log("train_vae_loss", vae_loss)
        return vae_loss


    def validation_step(self, batch, batch_idx):
        x = batch['s1']['xt'].reshape(-1, self.input_dim)
        x_recon, mu, logvar = self.net(x)
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
        total_kld, dimension_wise_kld, mean_kld = kl_divergence(mu, logvar)
        vae_loss = recon_loss + self.beta * total_kld        
        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mu.view(-1, self.z_dim).T.detach().cpu().numpy()
        if "yt" in batch['s1']:
            zt_true = batch['s1']["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
            mcc = compute_mcc(zt_recon, zt_true, self.correlation)
            self.log("val_mcc", mcc) 
            
        self.log("val_vae_loss", vae_loss)
        return vae_loss

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), 
                                 lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=0.0001)
        return opt_v

class BetaKittiConv(pl.LightningModule):
    def __init__(self, 
                 nc, 
                 z_dim, 
                 hidden_dim, 
                 beta,
                 lr,
                 correlation,
                 decoder_dist='bernoulli'):
        # Networks & Optimizers
        super(BetaKittiConv, self).__init__()
        self.beta = beta
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.correlation = correlation
        self.decoder_dist = decoder_dist
        self.lr = lr
        self.net = BetaVAE_CNN(nc=nc, 
                               z_dim=z_dim,
                               hidden_dim=hidden_dim)
    
    def training_step(self, batch, batch_idx):
        _, _, nc, h, w = batch['s1']['xt'].shape
        x = batch['s1']['xt'].reshape(-1, nc, h, w)
        x_recon, mu, logvar = self.net(x)
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
        total_kld, dimension_wise_kld, mean_kld = kl_divergence(mu, logvar)
        vae_loss = recon_loss + self.beta * total_kld
                
        self.log("train_vae_loss", vae_loss)
        return vae_loss


    def validation_step(self, batch, batch_idx):
        batch_size, length, nc, h, w = batch['s1']['xt'].shape
        x = batch['s1']['xt'].reshape(-1, nc, h, w)
        x_recon, mu, logvar = self.net(x)
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
        total_kld, dimension_wise_kld, mean_kld = kl_divergence(mu, logvar)
        vae_loss = recon_loss + self.beta * total_kld        
        # Compute Mean Correlation Coefficient (MCC)
        mus = mu.reshape(batch_size, length, self.z_dim)
        zt_recon = mus[:,-1,:].T.detach().cpu().numpy()
        zt_true = batch['s1']["yt"][:,-1,:].squeeze().T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)

        self.log("val_mcc", mcc) 
        self.log("val_vae_loss", vae_loss)
        return vae_loss

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), 
                                 lr=self.lr, weight_decay=0.0001)
        return opt_v

class BetaBallKeypoint(pl.LightningModule):
    def __init__(self, 
                 nc, 
                 k,
                 beta,
                 lr,
                 correlation,
                 nf=16,
                 decoder_dist='gaussian'):
        # Networks & Optimizers
        super().__init__()
        self.beta = beta
        self.z_dim = k*2
        self.correlation = correlation
        self.decoder_dist = decoder_dist
        self.lr = lr
        self.net = BetaVAE_KP(k=k, 
                              nc=nc, 
                              nf=nf, 
                              norm_layer='Batch')
    
    def training_step(self, batch, batch_idx):
        batch_size, length, nc, h, w = batch['s1']['xt'].shape
        x, y, ct = batch['s1']['xt'], batch['s1']['yt'], batch['s1']['ct']
        x_recon, mu, logvar, z = self.net(x.view(-1, nc, h, w), return_z=True)
        x_recon = x_recon.reshape(batch_size, length, nc, h, w)
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
        total_kld, dimension_wise_kld, mean_kld = kl_divergence(mu, logvar)
        vae_loss = recon_loss + self.beta * total_kld
                
        self.log("train_vae_loss", vae_loss)
        return vae_loss


    def validation_step(self, batch, batch_idx):
        batch_size, length, nc, h, w = batch['s1']['xt'].shape
        x, y, ct = batch['s1']['xt'], batch['s1']['yt'], batch['s1']['ct']
        x_recon, mu, logvar, z = self.net(x.view(-1, nc, h, w), return_z=True)
        x_recon = x_recon.reshape(batch_size, length, nc, h, w)
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
        total_kld, dimension_wise_kld, mean_kld = kl_divergence(mu, logvar)
        vae_loss = recon_loss + self.beta * total_kld        
        # Compute Mean Correlation Coefficient (MCC)
        mus = mu.reshape(batch_size, length, self.z_dim)
        zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()
        zt_true = batch['s1']["yt"][...,:2].reshape(batch_size,length,-1).view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)

        self.log("val_mcc", mcc) 
        self.log("val_vae_loss", vae_loss)
        return vae_loss

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), 
                                 lr=self.lr, weight_decay=0.0001)
        return opt_v