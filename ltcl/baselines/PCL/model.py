
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from .mix import (MixingMLP, ScoringFunc)
from ..metrics.correlation import compute_mcc

import ipdb as pdb

class PCL(pl.LightningModule):

    def __init__(self, 
                 z_dim, 
                 lags=2, 
                 hidden_dims=64, 
                 encoder_layers=3, 
                 scoring_layers=3,
                 correlation='Pearson',
                 lr=0.001):
        super().__init__()
        self.z_dim = z_dim
        self.L = lags
        self.lr = lr
        self.encoder = MixingMLP(input_dims=z_dim, 
                                 num_layers=encoder_layers, 
                                 negative_slope=0.2)

        self.scoring_funcs = nn.ModuleList([
            ScoringFunc(input_dims=lags+1, 
                        hidden_dims=hidden_dims, 
                        num_layers=scoring_layers) for _ in range(z_dim)]
            )

        self.loss_func= F.binary_cross_entropy_with_logits
        self.correlation = correlation

    def forward(self, x):
        embeddings = [ ]
        for sample in x:
            embeddings.append(self.encoder(x))
        return embeddings
    
    def training_step(self, batch, batch_idx):
        # x_pos: [BS, L+1 , D]
        x_pos, x_neg = batch['pos']['x'], batch['neg']['x']
        x_pos = x_pos.view(-1, self.L+1, self.z_dim)
        x_neg = x_neg.view(-1, self.L+1, self.z_dim)
        batch_size = x_pos.shape[0]
        cat = torch.cat((x_pos, x_neg), dim=0)
        embeddings = self.encoder(cat) # [2BS, L+1 , D]
        # embeddings of shape BS X sources X contrastive_samples
        scores = 0
        for i in range(self.z_dim):
            embedding = embeddings[:,:,i]
            score = self.scoring_funcs[i](embedding)
            scores = scores + score
        scores = scores.squeeze()
        ones = torch.ones(batch_size, device=x_pos.device)
        zeros = torch.zeros(batch_size, device=x_pos.device)
        loss = 0.5 * (self.loss_func(scores[:batch_size], ones) + self.loss_func(scores[batch_size:], zeros))
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['s1']['xt']
        embeddings = self.encoder(x)
        zt_recon = embeddings.view(-1, self.z_dim).T.detach().cpu().numpy()
        zt_true = batch['s1']["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)
        self.log("val_mcc", mcc) 
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer