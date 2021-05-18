import os
import torch
import torchvision
import numpy as np
from tqdm import trange
from sklearn import preprocessing
from torch.nn import functional as F

import sys 
sys.path.append("..") 
from utils.utils_vae import compute_mcc, compute_mmd

standard_scaler = preprocessing.StandardScaler()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loss_func(et_mean, et_logvar, recon_xt_, xt_, mask_matrix):
	"""
	Computes the VAE loss function.
	including:
		recon_loss = mse{recon_xt_, xt_}
		kld_loss = kl_divergence{et, gaussian}
		L1_loss for sparsity
	I suggest that we can add some coefficients before each part.
	"""
	batch_size = xt_.size(0)
	recons_loss = F.mse_loss(recon_xt_, xt_, reduction='sum')
	kld_loss = torch.mean(-0.5 * torch.sum(1 + et_logvar - et_mean ** 2 - et_logvar.exp(), dim = 1), dim = 0)
	l1_loss = torch.mean(torch.abs(mask_matrix))
	vae_loss = (recons_loss + kld_loss + l1_loss)/batch_size

	return vae_loss, recons_loss/batch_size, kld_loss/batch_size, l1_loss/batch_size

###### training model ######
class Trainer(object):
	def __init__(self, model, train_dataloader, test_dataloader, e_dim, learning_rate=1e-4, epochs=None, ckpoint='../ckpoint/vae_model.pth', recon_path='../recon/'):
		self.e_dim = e_dim
		self.epochs = epochs
		self.epoch_loss = []
		self.ckpoint = ckpoint
		self.recon_path = recon_path
		self.model = model.to(device)
		self.learning_rate = learning_rate
		self.test_dataloader = test_dataloader
		self.train_dataloader = train_dataloader
		self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)

	def save_checkpoint(self,epoch):
		torch.save({
			'epoch' : epoch + 1,
			'state_dict' : self.model.state_dict(),
			'optimizer' : self.optimizer.state_dict(),
			'losses' : self.epoch_loss},
			self.ckpoint)

	def load_checkpoint(self):
		try:
			print("Loading Checkpoint from '{}'".format(self.ckpoint))
			checkpoint = torch.load(self.ckpoint)
			self.start_epoch = checkpoint['epoch']
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			self.epoch_loss = checkpoint['losses']
			print("Resuming Training From Epoch {}".format(self.start_epoch))
		except:
			print("No Checkpoint Exists At '{}'.Start Fresh Training".format(self.ckpoint))
			self.start_epoch = 0

	def recon_frame(self, epoch, xt, xt_):
		with torch.no_grad():
			_, _, _, recon_xt_, _ = self.model.forward(xt, xt_) 
			BS = xt_.shape[0]
			image_ = torch.cat((xt_,recon_xt_),dim=0)
			os.makedirs(os.path.dirname('%s/epoch%d.png' % (self.recon_path,epoch)), exist_ok=True)
			torchvision.utils.save_image(image_,'%s/epoch_%d.png' % (self.recon_path, epoch))

	def val_model(self):
		# for sythetic data
		self.model.eval()
		val_loss = []
		for batch_sample in self.test_dataloader:
			XT = batch_sample["xt"]
			XT_ = batch_sample["xt_"]
			for i in range(XT.size()[1]):
				xt = XT[:,i,:]; xt_ = XT_[:,i,:]
				# xt = x_t        (batch_size, length, size)
				# xt_ = x_(t+1)   (batch_size, length, size)
				xt = torch.from_numpy(standard_scaler.fit_transform(xt)).to(device).float()
				xt_ = torch.from_numpy(standard_scaler.fit_transform(xt_)).to(device).float()
				et_mean, et_logvar, yt_, recon_xt_, mask_matrix = self.model.forward(xt, xt_)
				vae_loss, _, _, _ = loss_func(et_mean, et_logvar, recon_xt_, xt_, mask_matrix)
				val_loss.append(vae_loss.item())
		val_meanloss = np.mean(val_loss)
		self.model.train()
		return val_meanloss

	def train_model(self):
		self.model.train()
		l1_list = []
		mmd_list = []
		corr_list = []
		meanloss_list = []
		val_meanloss_list = []
		for epoch in trange(self.start_epoch, self.epochs):
			losses = []; recons = []; klds = []; l1s = []; corrs = []; mmds = []
			print("Running Epoch : {}".format(epoch+1))
			for batch_sample in self.train_dataloader:
				XT = batch_sample["xt"]
				XT_ = batch_sample["xt_"]
				YT_ = batch_sample["yt_"]
				for i in range(XT.size()[1]):
					xt = XT[:,i,:]; xt_ = XT_[:,i,:]
					xt = torch.from_numpy(standard_scaler.fit_transform(xt)).to(device).float()
					xt_ = torch.from_numpy(standard_scaler.fit_transform(xt_)).to(device).float()
					et_mean, et_logvar, yt_, recon_xt_, mask_matrix = self.model.forward(xt, xt_)
					vae_loss, recons_loss, kld_loss, l1_loss = loss_func(et_mean, et_logvar, recon_xt_, xt_, mask_matrix)
					losses.append(vae_loss.item())
					recons.append(recons_loss.item())
					klds.append(kld_loss.item())
					l1s.append(l1_loss.item())

					yt_np_ = yt_.cpu().detach().numpy()
					yt_real_ = YT_[:,i,:].cpu().detach().numpy()
					_, meanabscorr = compute_mcc(np.transpose(yt_np_), np.transpose(yt_real_), correlation_fn='Spearman')
					corrs.append(meanabscorr)

					mmd_loss = compute_mmd(yt_, YT_[:,i,:].to(device)).cpu().detach().numpy()
					mmds.append(mmd_loss)

					self.optimizer.zero_grad()
					vae_loss.backward()
					self.optimizer.step()

			meanloss = np.mean(losses); meanrecon = np.mean(recons)
			meankld = np.mean(klds); meanl1 = np.mean(l1s)
			meancorr = np.mean(corrs); meanmmd = np.mean(mmds)
			print("Epoch {}: vae loss: {}, recon loss: {}, kld loss: {}, l1 loss: {}, corr loss: {}, mmd loss: {}".format(epoch+1, meanloss, meanrecon, meankld, meanl1, meancorr, meanmmd))
			self.save_checkpoint(epoch)

			l1_list.append(meanl1)
			meanloss_list.append(meanloss)
			val_meanloss = self.val_model()
			val_meanloss_list.append(val_meanloss)
			corr_list.append(meancorr); mmd_list.append(meanmmd)

		np.save("l1_loss.npy", np.asarray(l1_list))
		np.save("corr_loss.npy", np.asarray(corr_list))
		np.save("mmd_loss.npy", np.asarray(mmd_list))
		np.save("training_loss.npy", np.asarray(meanloss_list))
		np.save("testing_loss.npy", np.asarray(val_meanloss_list))
