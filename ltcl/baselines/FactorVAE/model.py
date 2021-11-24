"""
FactorVAE.py
- https://github.com/1Konny/FactorVAE
- Contrastive Learning & beta-VAE
- No conditional information
"""
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

"utils file (SAME)"
from ..metrics.correlation import compute_mcc
"FactorVAE list"
from .net import FactorVAEMLP, FactorVAECNN, FactorVAEKP, Discriminator
from .ops import recon_loss, kl_divergence, permute_dims
import ipdb as pdb

class FactorVAE(pl.LightningModule):
    def __init__(self, 
                 input_dim, 
                 z_dim, 
                 hidden_dim, 
                 gamma, 
                 lr_VAE, 
                 beta1_VAE, 
                 beta2_VAE, 
                 lr_D, 
                 beta1_D, 
                 beta2_D, 
                 correlation):
        # Networks & Optimizers
        super(FactorVAE, self).__init__()
        self.z_dim = z_dim
        self.gamma = gamma
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.correlation = correlation
        self.lr_VAE = lr_VAE
        self.beta1_VAE = beta1_VAE
        self.beta2_VAE = beta2_VAE

        self.lr_D = lr_D
        self.beta1_D = beta1_D
        self.beta2_D = beta2_D

        self.VAE = FactorVAEMLP(self.input_dim, self.z_dim, self.hidden_dim)
        self.D = Discriminator(self.z_dim, self.hidden_dim)


    def training_step(self, batch, batch_idx, optimizer_idx):
        x_true1 = batch['s1']['xt'].reshape(-1, self.input_dim)
        x_true2 = batch['s2']['xt'].reshape(-1, self.input_dim)
        batch_size = x_true1.shape[0]
        ones = torch.ones(batch_size, dtype=torch.long, device=x_true1.device)
        zeros = torch.zeros(batch_size, dtype=torch.long, device=x_true1.device)

        x_recon, mu, logvar, z = self.VAE(x_true1)
        vae_recon_loss = recon_loss(x_true1, x_recon)
        vae_kld = kl_divergence(mu, logvar)

        D_z = self.D(z)
        vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

        # VAE training
        if optimizer_idx == 0:
            vae_loss = vae_recon_loss + vae_kld + self.gamma*vae_tc_loss
            self.log("train_vae_loss", vae_loss)
            self.log("train_vae_recon_loss", vae_recon_loss)
            self.log("train_vae_kld", vae_kld)
            self.log("train_vae_tc_loss", vae_tc_loss)
            return vae_loss

        # Discriminator training
        if optimizer_idx == 1:  
            z_prime = self.VAE(x_true2, no_dec=True)
            z_pperm = permute_dims(z_prime).detach()
            D_z_pperm = self.D(z_pperm)
            D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))
            self.log("train_d_tc_loss", D_tc_loss)
            return D_tc_loss

    def validation_step(self, batch, batch_idx):
        x_true1 = batch['s1']['xt'].reshape(-1, self.input_dim)

        x_recon, mu, logvar, z = self.VAE(x_true1)
        vae_recon_loss = recon_loss(x_true1, x_recon)
        vae_kld = kl_divergence(mu, logvar)

        D_z = self.D(z)
        vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

        vae_loss = vae_recon_loss + vae_kld + self.gamma*vae_tc_loss

        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mu.view(-1, self.z_dim).T.detach().cpu().numpy()
        if "yt" in batch['s1']:
            zt_true = batch['s1']["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
            mcc = compute_mcc(zt_recon, zt_true, self.correlation)
            self.log("val_mcc", mcc) 
            
        self.log("val_vae_loss", vae_loss)
        self.log("val_vae_recon_loss", vae_recon_loss)
        self.log("val_vae_kld", vae_kld)
        self.log("val_vae_tc_loss", vae_tc_loss)
        return vae_loss

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.VAE.parameters()), 
                                 lr=self.lr_VAE, betas=(self.beta1_VAE, self.beta2_VAE), weight_decay=0.0001)
        opt_d = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), 
                                 lr=self.lr_D, betas=(self.beta1_D, self.beta2_D))
        return [opt_v, opt_d], []

class FactorVAEKitti(pl.LightningModule):
    def __init__(self, 
                 nc, 
                 z_dim, 
                 hidden_dim, 
                 gamma, 
                 lr_VAE, 
                 lr_D, 
                 correlation,
                 decoder_dist='bernoulli'):
        # Networks & Optimizers
        super(FactorVAEKitti, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.correlation = correlation
        self.lr_VAE = lr_VAE
        self.lr_D = lr_D
        self.decoder_dist = decoder_dist

        self.VAE = FactorVAECNN(nc=nc, 
                                z_dim=z_dim,
                                hidden_dim=hidden_dim)
                               
        self.D = Discriminator(self.z_dim, self.hidden_dim)

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch_size, length, nc, h, w = batch['s1']['xt'].shape
        x_true1 = batch['s1']['xt'].reshape(-1, nc, h, w)
        x_true2 = batch['s2']['xt'].reshape(-1, nc, h, w)
        ones = torch.ones(batch_size*length, dtype=torch.long, device=x_true1.device)
        zeros = torch.zeros(batch_size*length, dtype=torch.long, device=x_true1.device)

        x_recon, mu, logvar, z = self.VAE(x_true1)
        vae_recon_loss = recon_loss(x_true1, x_recon, decoder_dist=self.decoder_dist)
        vae_kld = kl_divergence(mu, logvar)

        D_z = self.D(z)
        vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

        # VAE training
        if optimizer_idx == 0:
            vae_loss = vae_recon_loss + vae_kld + self.gamma*vae_tc_loss
            self.log("train_vae_loss", vae_loss)
            self.log("train_vae_recon_loss", vae_recon_loss)
            self.log("train_vae_kld", vae_kld)
            self.log("train_vae_tc_loss", vae_tc_loss)
            return vae_loss

        # Discriminator training
        if optimizer_idx == 1:  
            z_prime = self.VAE(x_true2, no_dec=True)
            z_pperm = permute_dims(z_prime).detach()
            D_z_pperm = self.D(z_pperm)
            D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))
            self.log("train_d_tc_loss", D_tc_loss)
            return D_tc_loss

    def validation_step(self, batch, batch_idx):
        batch_size, length, nc, h, w = batch['s1']['xt'].shape
        x_true1 = batch['s1']['xt'].reshape(-1, nc, h, w)

        x_recon, mu, logvar, z = self.VAE(x_true1)
        vae_recon_loss = recon_loss(x_true1, x_recon, decoder_dist=self.decoder_dist)
        vae_kld = kl_divergence(mu, logvar)

        D_z = self.D(z)
        vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

        vae_loss = vae_recon_loss + vae_kld + self.gamma*vae_tc_loss

        # Compute Mean Correlation Coefficient (MCC)
        mus = mu.reshape(batch_size, length, self.z_dim)
        zt_recon = mus[:,-1,:].T.detach().cpu().numpy()
        zt_true = batch['s1']["yt"][:,-1,:].squeeze().T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)

        self.log("val_mcc", mcc) 
        self.log("val_vae_loss", vae_loss)
        self.log("val_vae_recon_loss", vae_recon_loss)
        self.log("val_vae_kld", vae_kld)
        self.log("val_vae_tc_loss", vae_tc_loss)
        return vae_loss

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.VAE.parameters()), 
                                 lr=self.lr_VAE, weight_decay=0.0001)
        opt_d = torch.optim.SGD(filter(lambda p: p.requires_grad, self.D.parameters()), 
                                 lr=self.lr_D)
        return [opt_v, opt_d], []

class FactorBallKeypoint(pl.LightningModule):
    def __init__(self, 
                 k, 
                 nc, 
                 nf, 
                 gamma, 
                 hidden_dim,
                 lr_VAE, 
                 lr_D, 
                 correlation,
                 decoder_dist='bernoulli'):
        # Networks & Optimizers
        super().__init__()
        self.nc = nc
        self.k = k
        self.nf = nf
        self.z_dim = self.k * 2 
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.correlation = correlation
        self.lr_VAE = lr_VAE
        self.lr_D = lr_D
        self.decoder_dist = decoder_dist

        self.VAE = FactorVAEKP(k=k, nc=nc, nf=nf)
        self.D = Discriminator(self.z_dim, self.hidden_dim)

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch_size, length, nc, h, w = batch['s1']['xt'].shape
        x_true1 = batch['s1']['xt'].reshape(-1, nc, h, w)
        x_true2 = batch['s2']['xt'].reshape(-1, nc, h, w)
        ones = torch.ones(batch_size*length, dtype=torch.long, device=x_true1.device)
        zeros = torch.zeros(batch_size*length, dtype=torch.long, device=x_true1.device)

        x_recon, mu, logvar, z = self.VAE(x_true1)
        vae_recon_loss = recon_loss(x_true1, x_recon, decoder_dist=self.decoder_dist)
        vae_kld = kl_divergence(mu, logvar)

        D_z = self.D(z)
        vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

        # VAE training
        if optimizer_idx == 0:
            vae_loss = vae_recon_loss + vae_kld + self.gamma*vae_tc_loss
            self.log("train_vae_loss", vae_loss)
            self.log("train_vae_recon_loss", vae_recon_loss)
            self.log("train_vae_kld", vae_kld)
            self.log("train_vae_tc_loss", vae_tc_loss)
            return vae_loss

        # Discriminator training
        if optimizer_idx == 1:  
            z_prime = self.VAE(x_true2, no_dec=True)
            z_pperm = permute_dims(z_prime).detach()
            D_z_pperm = self.D(z_pperm)
            D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))
            self.log("train_d_tc_loss", D_tc_loss)
            return D_tc_loss

    def validation_step(self, batch, batch_idx):
        batch_size, length, nc, h, w = batch['s1']['xt'].shape
        x_true1 = batch['s1']['xt'].reshape(-1, nc, h, w)

        x_recon, mu, logvar, z = self.VAE(x_true1)
        vae_recon_loss = recon_loss(x_true1, x_recon, decoder_dist=self.decoder_dist)
        vae_kld = kl_divergence(mu, logvar)

        D_z = self.D(z)
        vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

        vae_loss = vae_recon_loss + vae_kld + self.gamma*vae_tc_loss

        # Compute Mean Correlation Coefficient (MCC)
        mus = mu.reshape(batch_size, length, self.z_dim)
        zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()
        zt_true = batch['s1']["yt"][...,:2].reshape(batch_size,length,-1).view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)

        self.log("val_mcc", mcc) 
        self.log("val_vae_loss", vae_loss)
        self.log("val_vae_recon_loss", vae_recon_loss)
        self.log("val_vae_kld", vae_kld)
        self.log("val_vae_tc_loss", vae_tc_loss)
        return vae_loss

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.VAE.parameters()), 
                                 lr=self.lr_VAE, weight_decay=0.0001)
        opt_d = torch.optim.SGD(filter(lambda p: p.requires_grad, self.D.parameters()), 
                                 lr=self.lr_D)
        return [opt_v, opt_d], []