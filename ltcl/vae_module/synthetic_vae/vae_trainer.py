import torch
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn import preprocessing
from torch.nn import functional as F

from vae_model import TemporalVAE
from utils_vae import SimulationDataset, compute_mcc, compute_mmd

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
	def __init__(self, model, train_dataloader, test_dataloader, learning_rate=1e-4, epochs=None, ckpoint='./ckpoint/vae_model.pth'):
		self.beta = 1
		self.z_dim = 2
		self.epochs = epochs
		self.epoch_loss = []
		self.ckpoint = ckpoint
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

	def val_model(self):
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
			for batch_sample in self.train_dataloader:
				XT = batch_sample["xt"]
				XT_ = batch_sample["xt_"]
				YT_ = batch_sample["yt_"]
				for i in range(XT.size()[1]):
					xt = XT[:,i,:]; xt_ = XT_[:,i,:]
					xt = torch.from_numpy(standard_scaler.fit_transform(xt)).to(device).float()
					xt_ = torch.from_numpy(standard_scaler.fit_transform(xt_)).to(device).float()
					yt_mean, yt_logvar, yt, yt_, y_mean_prior, y_logvar_prior, recon_xt, recon_xt_ = self.model.forward(xt, xt_)
					vae_loss, recons_loss, kld_loss = loss_func(xt, xt_, recon_xt, recon_xt_, self.beta, yt_mean, yt_logvar, y_mean_prior, y_logvar_prior)
					losses.append(vae_loss.item())
					recons.append(recons_loss.item())
					klds.append(kld_loss.item())

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
			meankld = np.mean(klds)
			meancorr = np.mean(corrs); meanmmd = np.mean(mmds)
			print("Epoch {}: vae loss: {}, recon loss: {}, kld loss: {}, corr loss: {}, mmd loss: {}".format(epoch+1, meanloss, meanrecon, meankld, meancorr, meanmmd))
			self.save_checkpoint(epoch)

			meanloss_list.append(meanloss)
			val_meanloss = self.val_model()
			val_meanloss_list.append(val_meanloss)
			corr_list.append(meancorr); mmd_list.append(meanmmd)
		np.save("training_loss.npy", np.asarray(meanloss_list))
		np.save("testing_loss.npy", np.asarray(val_meanloss_list))
		np.save("corr_loss.npy", np.asarray(corr_list))
		np.save("mmd_loss.npy", np.asarray(mmd_list))

###### training realization ######
test_dataset = SimulationDataset("val")
train_dataset = SimulationDataset("train")
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
vae_model = TemporalVAE(y_dim=2, input_dim=2, hidden_dim=2)
trainer = Trainer(vae_model, train_dataloader, test_dataloader, epochs=30)
torch.cuda.empty_cache()
trainer.load_checkpoint()
trainer.train_model()

#%%
###### test ######
# test the distribution of noise distribution
for batch_sample in test_dataloader:
	XT = batch_sample["xt"]
	XT_ = batch_sample["xt_"]
	YT = batch_sample["yt"]
	YT_ = batch_sample["yt_"]
	for i in range(XT.size()[1]):
		xt = XT[:,i,:]; xt_ = XT_[:,i,:]
		yt = YT[:,i,:]; yt_ = YT_[:,i,:]; 

		xt = torch.from_numpy(standard_scaler.fit_transform(xt)).to(device).float()
		xt_ = torch.from_numpy(standard_scaler.fit_transform(xt_)).to(device).float()
		yt_mean, yt_logvar, yt_estimate, yt_estimate_, y_mean_prior, y_logvar_prior, recon_xt, recon_xt_ = vae_model.forward(xt, xt_)

		yt_truth_np_ = yt_.cpu().detach().numpy()
		yt_estimate_np_ = yt_estimate_.cpu().detach().numpy()
		print("yt_truth is:", yt_truth_np_.shape) # (3200, 2)
		print("yt_estimated is:", yt_estimate_np_.shape) # (3200, 2)
		break
	break

#%%
# yt
plt.figure(figsize=(10,5))
plt.suptitle('yt', fontsize=14)
plt.subplot(221)
plt.title('estimated')
plt.hist(yt_estimate_np_[:,0])
plt.hist(yt_estimate_np_[:,1])
plt.subplot(222)
plt.title('ground_truth')
plt.hist(yt_truth_np_[:,0])
plt.hist(yt_truth_np_[:,1])

# test the amari_distancc(M, M_)
# plot the loss curve of train_dataloader/test_dataloader
mmd_loss_curve = np.load("mmd_loss.npy")
corr_loss_curve = np.load("corr_loss.npy")
test_loss_curve = np.load("testing_loss.npy")
train_loss_curve = np.load("training_loss.npy")

plt.figure(figsize=(20,5))
plt.subplot(141)
plt.plot(corr_loss_curve)
plt.title("corr_loss_curve")
plt.subplot(142)
plt.plot(mmd_loss_curve)
plt.title("mmd_loss_curve")
plt.subplot(143)
plt.plot(test_loss_curve)
plt.title("test_loss_curve")
plt.subplot(144)
plt.plot(train_loss_curve)
plt.title("train_loss_curve")
