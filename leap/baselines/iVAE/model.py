"""
iVAE.py
- https://github.com/ilkhem/icebeem
- Conditional --> Use CINS Dataset
- No Beta-VAE & Contrastive Learning
"""
import torch
import pytorch_lightning as pl

"utils file (SAME)"
from ..metrics.correlation import compute_mcc
"iVAE list"
from .net import iVAEMLP
import ipdb as pdb

class iVAE(pl.LightningModule):
    def __init__(self, 
                 input_dim, 
                 z_dim, 
                 hidden_dim, 
                 lr, 
                 correlation):
        # Networks & Optimizers
        super(iVAE, self).__init__()
        self.lr = lr
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.correlation = correlation
        self.model = iVAEMLP(self.input_dim, self.z_dim, self.hidden_dim)

    def training_step(self, batch, batch_idx):
        x = batch['s1']['xt']
        length = x.shape[1]
        x = x.reshape(-1, self.input_dim)
        index = batch['s1']['ct'].repeat_interleave(length).unsqueeze(-1)
        elbo, _ = self.model.elbo(x, index)
        vae_loss = elbo.mul(-1)
        self.log("train_vae_loss", vae_loss)
        return vae_loss

    def validation_step(self, batch, batch_idx):
        x = batch['s1']['xt']
        length = x.shape[1]
        x = x.reshape(-1, self.input_dim)
        index = batch['s1']['ct'].repeat_interleave(length).unsqueeze(-1)
        elbo, zt_recon = self.model.elbo(x, index)
        vae_loss = elbo.mul(-1)
        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = zt_recon.view(-1, self.z_dim).T.detach().cpu().numpy()
        if "yt" in  batch['s1']:
            zt_true = batch['s1']["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
            mcc = compute_mcc(zt_recon, zt_true, self.correlation)
            self.log("val_mcc", mcc) 
            
        self.log("val_vae_loss", vae_loss)
        return vae_loss

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        return opt_v
