import os
import torch
import torchvision
import numpy as np
from tqdm import trange
from sklearn import preprocessing
from torch.nn import functional as F

from utils_vae import compute_mcc, compute_mmd

standard_scaler = preprocessing.StandardScaler()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loss_func(xt, xt_, recon_xt, recon_xt_, beta, yt_mean, yt_logvar, y_mean_prior, y_logvar_prior):
	"""
	Computes the VAE loss function.
	including:
		recon_loss = recon{xt,xt~} + recon{xt_,xt_~}
		kld_loss = kl_divergence{yt, gaussian}
	"""
	batch_size = xt.size(0)
	recons_loss = F.mse_loss(recon_xt, xt, reduction='sum') + F.mse_loss(recon_xt_, xt_, reduction='sum')
	y_post_var = torch.exp(yt_logvar)
	y_prior_var = torch.exp(y_logvar_prior)
	kld_z = 0.5 * torch.sum(y_logvar_prior - yt_logvar + ((y_post_var + torch.pow(yt_mean - y_mean_prior, 2)) / y_prior_var) - 1)
	vae_loss = (recons_loss + beta * kld_z)/batch_size

	return vae_loss, recons_loss/batch_size, kld_z/batch_size

###### training model ######
class Trainer(object):
	def __init__(self, model, train_dataloader, test_dataloader, learning_rate=1e-4, epochs=None, ckpoint='./ckpoint/vae_model.pth', recon_path='./recon/'):
		self.beta = 1
		self.z_dim = 2
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
			_, _, _, _, _, _, recon_xt, recon_xt_ = self.model.forward(xt, xt_) 

# 			frame_index = int(torch.randint(0,xt.shape[0],(1,)).item())
# 			xt = xt[frame_index,:,:,:]
# 			xt_ = xt_[frame_index,:,:,:]
# 			recon_xt = recon_xt[frame_index,:,:,:]
# 			recon_xt_ = recon_xt_[frame_index,:,:,:]

			image = torch.cat((xt,recon_xt),dim=0)
			image_ = torch.cat((xt_,recon_xt_),dim=0)
			image = image.view(2*16,3,64,64)
			image_ = image_.view(2*16,3,64,64)
			os.makedirs(os.path.dirname('%s/epoch%d.png' % (self.recon_path,epoch)), exist_ok=True)
			torchvision.utils.save_image(image,'%s/epoch%d.png' % (self.recon_path, epoch))
			torchvision.utils.save_image(image_,'%s/epoch_%d.png' % (self.recon_path, epoch))

	def val_model(self):
		self.model.eval()
		val_loss = []
		for xt, xt_, _, _ in self.test_dataloader:
			xt = xt.to(device); xt_ = xt_.to(device)
			yt_mean, yt_logvar, yt, yt_, y_mean_prior, y_logvar_prior, recon_xt, recon_xt_ = self.model.forward(xt, xt_)
			vae_loss, _, _ = loss_func(xt, xt_, recon_xt, recon_xt_, self.beta, yt_mean, yt_logvar, y_mean_prior, y_logvar_prior)
			val_loss.append(vae_loss.item())
		val_meanloss = np.mean(val_loss)
		self.model.train()
		return val_meanloss

	def train_model(self):
		self.model.train()
		mmd_list = []
		corr_list = []
		meanloss_list = []
		val_meanloss_list = []
		for epoch in trange(self.start_epoch, self.epochs):
			losses = []; recons = []; klds = []; corrs = []; mmds = []
			print("Running Epoch : {}".format(epoch+1))
			for xt, xt_, _, _ in self.train_dataloader:
				xt = xt.to(device); xt_ = xt_.to(device)
				yt_mean, yt_logvar, yt, yt_, y_mean_prior, y_logvar_prior, recon_xt, recon_xt_ = self.model.forward(xt, xt_)
				vae_loss, recons_loss, kld_loss = loss_func(xt, xt_, recon_xt, recon_xt_, self.beta, yt_mean, yt_logvar, y_mean_prior, y_logvar_prior)
				losses.append(vae_loss.item())
				recons.append(recons_loss.item())
				klds.append(kld_loss.item())

				# yt_np_ = yt_.cpu().detach().numpy()
				# yt_real_ = YT_[:,i,:].cpu().detach().numpy()
				# _, meanabscorr = compute_mcc(np.transpose(yt_np_), np.transpose(yt_real_), correlation_fn='Spearman')
				# corrs.append(meanabscorr)

				# mmd_loss = compute_mmd(yt_, YT_[:,i,:].to(device)).cpu().detach().numpy()
				# mmds.append(mmd_loss)

				self.optimizer.zero_grad()
				vae_loss.backward()
				self.optimizer.step()

			meanloss = np.mean(losses); meanrecon = np.mean(recons); meankld = np.mean(klds)
			# meancorr = np.mean(corrs); meanmmd = np.mean(mmds)
			print("Epoch {}: vae loss: {}, recon loss: {}, kld loss: {}".format(epoch+1, meanloss, meanrecon, meankld))
			# print("Epoch {}: vae loss: {}, recon loss: {}, kld loss: {}, corr loss: {}, mmd loss: {}".format(epoch+1, meanloss, meanrecon, meankld, meancorr, meanmmd))
			self.save_checkpoint(epoch)

			self.model.eval()
			index = 0
			if epoch % 5 == 0:
				for xt, xt_, _, _ in self.test_dataloader:
					xt = xt.to(device); xt_ = xt_.to(device)
					index = index + 1
					if index == int(torch.randint(0,len(self.test_dataloader),(1,)).item()):
						self.recon_frame(epoch+1, xt, xt_)
						break

			meanloss_list.append(meanloss)
			val_meanloss = self.val_model()
			val_meanloss_list.append(val_meanloss)
			# corr_list.append(meancorr); mmd_list.append(meanmmd)
		np.save("training_loss.npy", np.asarray(meanloss_list))
		np.save("testing_loss.npy", np.asarray(val_meanloss_list))
		# np.save("corr_loss.npy", np.asarray(corr_list))
		# np.save("mmd_loss.npy", np.asarray(mmd_list))
