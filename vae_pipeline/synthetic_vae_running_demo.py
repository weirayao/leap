import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

from utils_vae import SimulationDataset
from synthetic_vae_model import TemporalVAE
from synthetic_vae_trainer import Trainer

standard_scaler = preprocessing.StandardScaler()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
		et_mean, et_logvar, yt_estimate_, recon_xt_, mask = vae_model.forward(xt, xt_)

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
l1_loss_curve = np.load("l1_loss.npy")
mmd_loss_curve = np.load("mmd_loss.npy")
corr_loss_curve = np.load("corr_loss.npy")
test_loss_curve = np.load("testing_loss.npy")
train_loss_curve = np.load("training_loss.npy")

plt.figure(figsize=(20,5))
plt.subplot(151)
plt.plot(corr_loss_curve)
plt.title("corr_loss_curve")
plt.subplot(152)
plt.plot(mmd_loss_curve)
plt.title("mmd_loss_curve")
plt.subplot(153)
plt.plot(test_loss_curve)
plt.title("test_loss_curve")
plt.subplot(154)
plt.plot(train_loss_curve)
plt.title("train_loss_curve")
plt.subplot(155)
plt.plot(l1_loss_curve)
plt.title("l1_loss_curve")
