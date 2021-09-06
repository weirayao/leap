"""
TCL.py
- pytorch-lightning version based on: https://github.com/ilkhem/icebeem
- Conditional --> Use CINS Dataset
- No Beta-VAE & Contrastive Learning
"""
import torch
import tensorflow as tf
import pytorch_lightning as pl

"utils file (SAME)"
from ..metrics.correlation import compute_mcc
"TCL list"
from .net import TCLMLP

def tcl_loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean


class TCL(pl.LightningModule):
    def __init__(self, 
                 input_dim, 
                 z_dim, 
                 nclass,
                 hidden_dim, 
                 lr, 
                 correlation):
        # Networks & Optimizers
        super(TCL, self).__init__()
        self.lr = lr
        self.z_dim = z_dim
        self.nclass = nclass
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.correlation = correlation
        self.model = TCLMLP(self.input_dim, self.z_dim, self.hidden_dim, self.nclass)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch['s1']['xt']; u = batch['s1']['ct']
        logits, feats = self.model(x)
        vae_loss = tcl_loss(logits, u)

        self.log("train_vae_loss", vae_loss)
        return vae_loss

    def validation_step(self, batch, batch_idx):
        x = batch['s1']['xt']; u = batch['s1']['ct']
        logits, feats = self.model(x)
        vae_loss = tcl_loss(logits.cpu(), u.cpu())
        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = feats.view(-1, self.z_dim).T.detach().cpu().numpy()
        zt_true = batch['s1']["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)

        self.log("val_mcc", mcc) 
        self.log("val_vae_loss", vae_loss)
        return vae_loss

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        return opt_v
