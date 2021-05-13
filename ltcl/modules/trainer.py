"""PyTorch LightningModule composed from modules"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .dendrogram import AfflineDendrogramFlow

class AffineFlow(pl.LightningModule):
    def __init__(self, cfg):
        pass

class PNLFlow(pl.LightningModule):
    def __init__(self, cfg):
        pass

class Dendrogram(pl.LightningModule):

    def __init__(
        self, 
        input_dims=(3,64,64),
        width = 64,
        depth = 16,
        n_levels = 3,
        lags = 4,
        lr = 1e-4,
        l1 = 0.01):
        super().__init__()
        self.lr = lr
        self.l1 = l1
        self.flow = AfflineDendrogramFlow(input_dims = input_dims, 
                                          width = width, 
                                          depth = depth, 
                                          n_levels = n_levels, 
                                          lags = lags)

        self.flow.load_state_dict(torch.load("/home/cmu_wyao/projects/dendrogram_init.pth"))


    def forward(self, imgs, past_imgs):
        # The forward function is only used for visualizing the graph
        return self.flow.log_prob(imgs, past_imgs)

    @torch.no_grad()
    def sample(self, past_imgs, z_init=None):
        """
        Sample a batch of next frame images from the flow.
        """
        # Sample latent representation from prior
        if z_init is None:
            z = self.prior.sample(sample_shape=img_shape).to(device)
        else:
            z = z_init.to(device)

        # Transform z to x by inverting the flows
        ldj = torch.zeros(img_shape[0], device=device)
        for flow in reversed(self.flows):
            z, ldj = flow(z, ldj, reverse=True)
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # An scheduler is optional, but can help in flows to get the last bpd improvement
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # Normalizing flows are trained by maximum likelihood => return bpd
        first_sample, second_sample, latents1, latents2 = batch
        log_probs = self.flow.log_prob(second_sample, first_sample)
        l1_regularization = 0
        for param in self.flow.topdown_kernel.parameters():
            l1_regularization = l1_regularization + torch.norm(param, 1)**2
        loss = -torch.mean(log_probs) + self.l1 * l1_regularization
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Normalizing flows are trained by maximum likelihood => return bpd
        first_sample, second_sample, latents1, latents2 = batch
        log_probs = self.flow.log_prob(second_sample, first_sample)
        l1_regularization = 0
        for param in self.flow.topdown_kernel.parameters():
            l1_regularization = l1_regularization + torch.norm(param, 1)**2
        loss = -torch.mean(log_probs) + self.l1 * l1_regularization
        self.log('val_loss', loss)
        return loss