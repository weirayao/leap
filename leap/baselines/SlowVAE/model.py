"""
SlowVAE.py
- https://github.com/bethgelab/slow_disentanglement
- Beta-VAE --> use Contrastive Learning Data
- No Contrastive Learning and Condition
"""
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

"utils file (SAME)"
from ..metrics.correlation import compute_mcc
"SlowVAE list"
from .net import SlowVAEMLP

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


class SlowVAE(pl.LightningModule):
    def __init__(self, 
                 input_dim, 
                 z_dim, 
                 hidden_dim, 
                 beta,
                 gamma, 
                 lr, 
                 beta1, 
                 beta2, 
                 rate_prior,
                 correlation):
        # Networks & Optimizers
        super(SlowVAE, self).__init__()
        self.beta = beta
        self.z_dim = z_dim
        self.gamma = gamma
        self.rate_prior = rate_prior
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.correlation = correlation
        self.decoder_dist = 'gaussian'
        self.rate_prior = rate_prior * torch.ones(1, requires_grad=False)

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.net = SlowVAEMLP(self.input_dim, self.z_dim, self.hidden_dim)

    def compute_cross_ent_laplace(self, mean, logvar, rate_prior):
        var = torch.exp(logvar)
        sigma = torch.sqrt(var)
        device = sigma.device
        rate_prior = rate_prior.to(device)
        normal_dist = torch.distributions.normal.Normal(
            torch.zeros(self.z_dim).to(device),
            torch.ones(self.z_dim).to(device))
        ce = - torch.log(rate_prior / 2) + rate_prior * sigma *\
             np.sqrt(2 / np.pi) * torch.exp(- mean**2 / (2 * var)) -\
             rate_prior * mean * (
                     1 - 2 * normal_dist.cdf(mean / sigma))
        return ce

    def compute_cross_ent_combined(self, mu, logvar):
        normal_entropy = compute_ent_normal(logvar)
        cross_ent_normal = compute_cross_ent_normal(mu, logvar)
        # assuming couples, do Laplace both ways
        mu0 = mu[::2]
        mu1 = mu[1::2]
        logvar0 = logvar[::2]
        logvar1 = logvar[1::2]
        rate_prior0 = self.rate_prior
        rate_prior1 = self.rate_prior
        cross_ent_laplace = (
            self.compute_cross_ent_laplace(mu0 - mu1, logvar0, rate_prior0) +
            self.compute_cross_ent_laplace(mu1 - mu0, logvar1, rate_prior1))
        return [x.sum(1).mean(0, True) for x in [normal_entropy,
                                                 cross_ent_normal,
                                                 cross_ent_laplace]]
    
    def training_step(self, batch, batch_idx):
        x = batch['s1']['xt']
        x_pst = x[:,:-1,:]
        x_cur = x[:,1:,:]
        cat = torch.cat((x_pst.reshape(-1, self.input_dim), 
                         x_cur.reshape(-1, self.input_dim)), 
                         dim=0)
        x_recon, mu, logvar = self.net(cat)
        recon_loss = reconstruction_loss(cat, x_recon, self.decoder_dist)

        # VAE training
        [normal_entropy, cross_ent_normal, cross_ent_laplace] = self.compute_cross_ent_combined(mu, logvar)
        vae_loss = 2 * recon_loss
        kl_normal = cross_ent_normal - normal_entropy
        kl_laplace = cross_ent_laplace - normal_entropy
        vae_loss = vae_loss + self.beta * kl_normal
        vae_loss = vae_loss + self.gamma * kl_laplace
                
        self.log("train_vae_loss", vae_loss)
        return vae_loss


    def validation_step(self, batch, batch_idx):
        x = batch['s1']['xt']
        x_pst = x[:,:-1,:]
        x_cur = x[:,1:,:]
        cat = torch.cat((x_pst.reshape(-1, self.input_dim), 
                         x_cur.reshape(-1, self.input_dim)), 
                         dim=0)
        x_recon, mu, logvar = self.net(cat)
        recon_loss = reconstruction_loss(cat, x_recon, self.decoder_dist)

        # VAE training
        [normal_entropy, cross_ent_normal, cross_ent_laplace] = self.compute_cross_ent_combined(mu, logvar)
        vae_loss = 2 * recon_loss
        kl_normal = cross_ent_normal - normal_entropy
        kl_laplace = cross_ent_laplace - normal_entropy
        vae_loss = vae_loss + self.beta * kl_normal
        vae_loss = vae_loss + self.gamma * kl_laplace
        
        # Compute Mean Correlation Coefficient (MCC)

        _, mu, _ = self.net(x.view(-1, self.input_dim))
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