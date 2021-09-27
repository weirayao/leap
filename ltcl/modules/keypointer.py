import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from .components.keypoint import KeyPointNet

import ipdb as pdb


class Keypointer(pl.LightningModule):

    def __init__(self, 
                 n_kps,
                 width=64,
                 height=64, 
                 nf=16, 
                 norm_layer='Batch', 
                 lim=[-1., 1., -1., 1.],
                 lr=1e-3):
                
        super().__init__()
        self.kp = KeyPointNet(k=n_kps, 
                              width=64, 
                              height=64, 
                              nf=16, 
                              norm_layer='Batch', 
                              lim=lim)
        self.lr = lr
    
    def forward(self, xt):
        batch_size, length, nc, h, w = xt.shape
        cat = xt.view(-1, nc, h, w)
        feat = self.kp.extract_feature(cat)
        kp = self.kp.predict_keypoint(cat)
        hmap = self.kp.keypoint_to_heatmap(kp)
        feat = feat.reshape(batch_size, length, feat.shape[1], feat.shape[2], feat.shape[3])
        kp = kp.reshape(batch_size, length, kp.shape[1], kp.shape[2])
        hmap = hmap.reshape(batch_size, length, hmap.shape[1], hmap.shape[2], hmap.shape[3])
        return feat, kp, hmap

    def reconstruct(self, kpts, feat, kptsr, featr):
        batch_size, length, n_kps, _ = kpts.shape
        cat = kpts.reshape(batch_size*length, n_kps, 2)
        catr = kptsr.reshape(batch_size*length, n_kps, 2)
        # Background is white 1.0
        hmap = self.kp.keypoint_to_heatmap(cat)
        hmap = hmap.reshape(batch_size, length, hmap.shape[1], hmap.shape[2], hmap.shape[3])
        hmapr = self.kp.keypoint_to_heatmap(catr)
        hmapr = hmapr.reshape(batch_size, length, hmapr.shape[1], hmapr.shape[2], hmapr.shape[3])       
        x_recon = [ ]
        for t in range(length):
            mixed_feat = self.kp.original_transport(featr[:,t], feat[:,t], hmapr[:,t], hmap[:,t])
            des_pred = self.kp.refine(mixed_feat)
            x_recon.append(des_pred)
        x_recon = torch.stack(x_recon, 1)
        return x_recon

    def training_step(self, batch, batch_idx):
        src, des = batch['src'], batch['des']
        des_pred, _, _ = self.kp.forward(src, des)
        loss = F.mse_loss(des_pred, des)
        self.log("train_loss", loss) 
        return loss

    def validation_step(self, batch, batch_idx):
        src, des = batch['src'], batch['des']
        des_pred, _, _ = self.kp.forward(src, des)
        loss = F.mse_loss(des_pred, des)
        self.log("val_loss", loss) 
        return loss
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        return opt
    

