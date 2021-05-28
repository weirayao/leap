"""Linear Latent Transition Causal Model Estimated by Temporal VAE"""
import torch
from torch import nn
import torch.nn.functional as F
from .components.vae import LinearTemporalVAESynthetic
from .components.tc import Discriminator, permute_dims
from .metrics.correlation import compute_mcc
import pytorch_lightning as pl
import random
import ipdb as pdb

class AfflineVAESynthetic(pl.LightningModule):


	def __init__(
		self, 
		input_dim,
		y_dim, 
		lag = 1,
		e_dim = 128,
		kl_coeff = 1,
		lr = 1e-4,
		diagonal = False,
		negative_slope = 0.2):
		'''
		please add your flow module
		self.flow = XX
		'''
		super().__init__()
		self.y_dim = y_dim
		self.vae = LinearTemporalVAESynthetic(input_dim = input_dim, 
											  y_dim = y_dim, 
											  lag = lag,
											  e_dim = e_dim,
											  diagonal = diagonal,
											  negative_slope = negative_slope,
											  kl_coeff = kl_coeff)
		self.lr = lr
	
	def forward(self, batch):
		return self.model(batch)
	
	def training_step(self, batch, batch_idx):
		_, _, _, _, xt_, recon_xt_, p, q, e, _ = self.vae.forward(batch)
		loss = self.vae.elbo_loss(xt_, recon_xt_, p, q, e)
		self.log("train_elbo_loss", loss)
		return loss
	
	def validation_step(self, batch, batch_idx):
		e_mean, e_logvar, yt, yt_, xt_, recon_xt_, p, q, e, eps = self.vae.forward(batch)
		loss = self.vae.elbo_loss(xt_, recon_xt_, p, q, e)

		yt_recon = yt_.squeeze().T.detach().cpu().numpy()
		yt_true = batch["yt_"].squeeze().T.detach().cpu().numpy()
		mcc = compute_mcc(yt_recon, yt_true, "Pearson")
		self.log("val_elbo_loss", loss)
		self.log("val_mcc", mcc)
		return loss
	
	def sample(self, xt):
		batch_size = xt.shape[0]
		e = torch.randn(batch_size, self.y_dim)
		eps, _ = self.vae.spline.inverse(e)
		ut = self.vae.predict_next_latent(xt)
		xt_ = self.vae.decode(eps, ut)
		return xt_, eps

	def reconstruct(self):
		return self.forward(batch)[5]

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
		# An scheduler is optional, but can help in flows to get the last bpd improvement
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
		return [optimizer], [scheduler]

class AfflineVAESyntheticContrast(pl.LightningModule):


	def __init__(
		self, 
		input_dim,
		y_dim, 
		lag = 1,
		e_dim = 128,
		kl_coeff = 1,
		gamma = 25,
		lr = 1e-4,
		diagonal = False,
		negative_slope = 0.2):
		'''
		please add your flow module
		self.flow = XX
		'''
		super().__init__()
		self.y_dim = y_dim
		self.vae = LinearTemporalVAESynthetic(input_dim = input_dim, 
											  y_dim = y_dim, 
											  lag = lag,
											  e_dim = e_dim,
											  diagonal = diagonal,
											  negative_slope = negative_slope,
											  kl_coeff = kl_coeff)
		self.discriminator = Discriminator(z_dim = y_dim)
		self.lr = lr
		self.gamma = gamma
	
	def forward(self, batch):
		return self.model(batch)
	
	def training_step(self, batch, batch_idx, optimizer_idx):
		batch1, batch2 = batch
		batch_size = len(batch1['yt_'])
		_, _, _, _, xt_, recon_xt_, p, q, e, _ = self.vae.forward(batch1)
		# VAE training
		if optimizer_idx == 0:
			elbo_loss = self.vae.elbo_loss(xt_, recon_xt_, p, q, e)
			D_z = self.discriminator(e)
			tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()
			vae_loss = elbo_loss + self.gamma * tc_loss
			self.log("vae_loss", vae_loss)
			return vae_loss

		# Discriminator training
		if optimizer_idx == 1:
			ones = torch.ones(batch_size, dtype=torch.long).to(batch1['yt_'].device)
			zeros = torch.zeros(batch_size, dtype=torch.long).to(batch1['yt_'].device)
			e = e.detach()
			D_e = self.discriminator(e)
			_, _, _, _, _, _, _, _, e_prime, _ = self.vae.forward(batch2)
			e_perm = permute_dims(e_prime).detach()
			D_e_pperm = self.discriminator(e_perm)
			if random.random() < 0.1:
				D_tc_loss = 0.5*(F.cross_entropy(D_e, ones) + F.cross_entropy(D_e_pperm, zeros))
			else:
				D_tc_loss = 0.5*(F.cross_entropy(D_e, zeros) + F.cross_entropy(D_e_pperm, ones))
			self.log("tc_loss", D_tc_loss)
			return D_tc_loss

	def validation_step(self, batch, batch_idx):
		batch = batch[0]
		e_mean, e_logvar, yt, yt_, xt_, recon_xt_, p, q, e, eps = self.vae.forward(batch)
		loss = self.vae.elbo_loss(xt_, recon_xt_, p, q, e)
		yt_recon = yt_.squeeze().T.detach().cpu().numpy()
		yt_true = batch["yt_"].squeeze().T.detach().cpu().numpy()
		mcc = compute_mcc(yt_recon, yt_true, "Pearson")
		self.log("val_elbo_loss", loss)
		self.log("val_mcc", mcc)
		return loss
	
	def sample(self, xt):
		batch_size = xt.shape[0]
		e = torch.randn(batch_size, self.y_dim)
		eps, _ = self.vae.spline.inverse(e)
		ut = self.vae.predict_next_latent(xt)
		xt_ = self.vae.decode(eps, ut)
		return xt_, eps

	def reconstruct(self, batch):
		return self.forward(batch)[5]

	def configure_optimizers(self):
		opt_v = torch.optim.Adam(filter(lambda p: p.requires_grad, self.vae.parameters()), lr=2e-4)

		opt_d = torch.optim.SGD(filter(lambda p: p.requires_grad, self.discriminator.parameters()), 
									lr=1e-4)

		return [opt_v, opt_d], []