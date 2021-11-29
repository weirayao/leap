
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from .mix import MixingMLP, MixingCNN, MixingKP, ScoringFunc
from ..metrics.correlation import compute_mcc

import ipdb as pdb

class PCL(pl.LightningModule):

    def __init__(self,
                 input_dim, 
                 z_dim, 
                 lags=2, 
                 hidden_dims=64, 
                 encoder_layers=3, 
                 scoring_layers=3,
                 correlation='Pearson',
                 lr=0.001):
        super().__init__()
        self.input_dim=input_dim
        self.z_dim = z_dim
        self.L = lags
        self.lr = lr
        self.encoder = MixingMLP(input_dims=input_dim,
                                 z_dim=self.z_dim,
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
        return self.encoder(x)
    
    def training_step(self, batch, batch_idx):
        # x_pos: [BS, L+1 , D]
        x_pos, x_neg = batch['pos']['x'], batch['neg']['x']
        x_pos = x_pos.view(-1, self.L+1, self.input_dim)
        x_neg = x_neg.view(-1, self.L+1, self.input_dim)
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
        if "yt" in batch['s1']:
            x = batch['s1']['xt']
            embeddings = self.encoder(x)
            zt_recon = embeddings.view(-1, self.z_dim).T.detach().cpu().numpy()
            zt_true = batch['s1']["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
            mcc = compute_mcc(zt_recon, zt_true, self.correlation)
            self.log("val_mcc", mcc)
        else:
            x_pos, x_neg = batch['pos']['x'], batch['neg']['x']
            x_pos = x_pos.view(-1, self.L+1, self.input_dim)
            x_neg = x_neg.view(-1, self.L+1, self.input_dim)
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
            self.log("val_loss", loss) 
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class PCLBallKP(pl.LightningModule):

    def __init__(self, 
                 nc,
                 k,
                 nf, 
                 lags=2, 
                 hidden_dims=256, 
                 scoring_layers=3,
                 correlation='Pearson',
                 lr=0.001):
        super().__init__()
        self.k = k
        self.z_dim = k * 2 
        self.L = lags
        self.lr = lr
        self.encoder = MixingKP(k=k, nc=nc, nf=nf)

        self.scoring_funcs = nn.ModuleList([
            ScoringFunc(input_dims=lags+1, 
                        hidden_dims=hidden_dims, 
                        num_layers=scoring_layers) for _ in range(self.z_dim)]
            )

        self.loss_func= F.binary_cross_entropy_with_logits
        self.correlation = correlation

    def forward(self, x):
        return self.encoder(x)
    
    def training_step(self, batch, batch_idx):
        # x_pos: [BS, L+1 , D]
        x_pos, x_neg = batch['pos']['x'], batch['neg']['x']
        _, _, _, nc, h, w = x_pos.shape
        x_pos = x_pos.view(-1, self.L+1, nc, h, w)
        x_neg = x_neg.view(-1, self.L+1, nc, h, w)
        batch_size = x_pos.shape[0]
        cat = torch.cat((x_pos, x_neg), dim=0)
        embeddings = self.encoder(cat.reshape(-1, nc, h, w)).reshape(2*batch_size, self.L+1,self.z_dim) # [2BS, L+1 , D]
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
        batch_size, length, nc, h, w = x.shape
        embeddings = self.encoder(x.view(-1, nc, h, w))
        zt_recon = embeddings.reshape(batch_size,length,-1).view(-1, self.z_dim).T.detach().cpu().numpy()
        if "yt" in batch['s1']["yt"]:
            zt_true = batch['s1']["yt"][...,:2].reshape(batch_size,length,-1).view(-1, self.k*2).T.detach().cpu().numpy()
            mcc = compute_mcc(zt_recon, zt_true, self.correlation)
            self.log("val_mcc", mcc) 
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class PCLKitti(pl.LightningModule):

    def __init__(self, 
                 nc,
                 k, 
                 lags=2, 
                 hidden_dims=256, 
                 scoring_layers=3,
                 correlation='Pearson',
                 lr=0.001):
        super().__init__()
        self.z_dim = k*2
        self.L = lags
        self.lr = lr
        self.encoder = MixingCNN(z_dim=self.z_dim, nc=nc, hidden_dim=hidden_dims)

        self.scoring_funcs = nn.ModuleList([
            ScoringFunc(input_dims=lags+1, 
                        hidden_dims=hidden_dims, 
                        num_layers=scoring_layers) for _ in range(self.z_dim)]
            )

        self.loss_func= F.binary_cross_entropy_with_logits
        self.correlation = correlation

    def forward(self, x):
        return self.encoder(x)
    
    def training_step(self, batch, batch_idx):
        # x_pos: [BS, L+1 , D]
        x_pos, x_neg = batch['pos']['x'], batch['neg']['x']
        _, _, nc, h, w = x_pos.shape
        x_pos = x_pos.view(-1, self.L+1, nc, h, w)
        x_neg = x_neg.view(-1, self.L+1, nc, h, w)
        batch_size = x_pos.shape[0]
        cat = torch.cat((x_pos, x_neg), dim=0)
        embeddings = self.encoder(cat.reshape(-1, nc, h, w)).reshape(2*batch_size, self.L+1,self.z_dim) # [2BS, L+1 , D]
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
        batch_size, length, nc, h, w = x.shape
        embeddings = self.encoder(x.view(-1, nc, h, w))
        zt_recon = embeddings.view(batch_size, length, self.z_dim)[:,-1,:].T.detach().cpu().numpy()
        zt_true = batch['s1']["yt"].view(batch_size, length, 3)[:,-1,:].squeeze().T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)
        self.log("val_mcc", mcc) 
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer