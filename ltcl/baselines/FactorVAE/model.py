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
from .net import FactorVAEMLP, Discriminator
from .ops import recon_loss, kl_divergence, permute_dims

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
        zt_true = batch['s1']["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)

        self.log("val_mcc", mcc) 
        self.log("val_vae_loss", vae_loss)
        self.log("val_vae_recon_loss", vae_recon_loss)
        self.log("val_vae_kld", vae_kld)
        self.log("val_vae_tc_loss", vae_tc_loss)
        return vae_loss

    def configure_optimizers(self):
        opt_v = torch.optim.Adam(filter(lambda p: p.requires_grad, self.VAE.parameters()), 
                                 lr=self.lr_VAE, betas=(self.beta1_VAE, self.beta2_VAE))
        opt_d = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), 
                                 lr=self.lr_D, betas=(self.beta1_D, self.beta2_D))
        return [opt_v, opt_d], []