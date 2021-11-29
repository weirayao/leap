"""
TCL.py
- pytorch-lightning version based on: https://github.com/ilkhem/icebeem
- Conditional --> Use CINS Dataset
- No Beta-VAE & Contrastive Learning
"""
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

"utils file (SAME)"
from ..metrics.correlation import compute_mcc
"TCL list"
from .net import TCLMLP
import ipdb as pdb

# import tensorflow as tf
# def tcl_loss(logits, labels):
#     # Calculate the average cross entropy loss across the batch.
#     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
#         labels=labels, logits=logits, name='cross_entropy_per_example')
#     cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#     return cross_entropy_mean

def tcl_loss(logits, labels):
    batch_size = labels.size(0)
    assert batch_size != 0
    recon_loss = F.cross_entropy(
        logits, labels, size_average=False).div(batch_size)

    return recon_loss

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

    def training_step(self, batch, batch_idx):
        x = batch['s1']['xt']
        length = x.shape[1]
        x = x.reshape(-1, self.input_dim)
        index = batch['s1']['ct'].repeat_interleave(length).squeeze().to(torch.long)
        logits, feats = self.model(x)
        vae_loss = tcl_loss(logits, index)

        self.log("train_loss", vae_loss)
        return vae_loss

    def validation_step(self, batch, batch_idx):
        x = batch['s1']['xt']
        length = x.shape[1]
        x = x.reshape(-1, self.input_dim)
        index = batch['s1']['ct'].repeat_interleave(length).squeeze().squeeze().to(torch.long)
        logits, feats = self.model(x)
        vae_loss = tcl_loss(logits, index)
        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = feats.view(-1, self.z_dim).T.detach().cpu().numpy()
        if "yt" in batch['s1']:
            zt_true = batch['s1']["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
            mcc = compute_mcc(zt_recon, zt_true, self.correlation)
            self.log("val_mcc", mcc) 
        self.log("val_loss", vae_loss)
        return vae_loss

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        return opt_v
