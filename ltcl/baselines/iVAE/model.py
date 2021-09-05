"""
iVAE.py
- https://github.com/ilkhem/icebeem
- ConditionalDataset --> Use Contrastive Dataset
- No contrastive learning & beta_VAE
"""
import torch
import pytorch_lightning as pl

"utils file (SAME)"
from ..metrics.correlation import compute_mcc
"iVAE list"
from .net import iVAEMLP

class iVAE(pl.LightningModule):
    def __init__(self, 
                 input_dim, 
                 z_dim, 
                 hidden_dim, 
                 lr, 
                 correlation):
        # Networks & Optimizers
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.correlation = correlation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.lr = lr
        self.model = iVAEMLP(self.input_dim, self.z_dim, self.hidden_dim)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch['s1']['xt']; u = batch['s1']['ct']
        elbo, z_est = self.model(x, u)
        vae_loss = elbo.mul(-1)
        self.log("train_vae_loss", vae_loss)
        return vae_loss

    def validation_step(self, batch, batch_idx, optimizer_idx):
        x = batch['s1']['xt']; u = batch['s1']['ct']
        elbo, zt_recon = self.model.elbo(x, u)
        vae_loss = elbo.mul(-1)
        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = zt_recon.view(-1, self.z_dim).T.detach().cpu().numpy()
        zt_true = batch['s1']["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)

        self.log("val_mcc", mcc) 
        self.log("val_vae_loss", vae_loss)
        return vae_loss

    def configure_optimizers(self):
        opt_v = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt_v
