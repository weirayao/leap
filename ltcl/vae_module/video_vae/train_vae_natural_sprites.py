import torch
from torch.utils.data import DataLoader, random_split

from vae_trainer import Trainer
from vae_model import TemporalVAE
from utils_dataset import Cars3D, Shapes3D, KittiMasks, NaturalSprites

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################## loading dataset #####################################
# 1. car3d
# dataset = Cars3D(prior='uniform', rate=1, k=-1)
# train_data, val_data = random_split(dataset, [10000, 7568])
# 2. shapes3d
# dataset = Shapes3D(prior='laplace', rate=1, k=-1)
# train_data, val_data = random_split(dataset, [400000, 80000])
# 3. natural_sprites
dataset = NaturalSprites()
train_data, test_data, val_data = random_split(dataset, [200, 794, 206800])
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False)

#%%
########################## training model #####################################
vae_model = TemporalVAE()
trainer = Trainer(vae_model, train_dataloader, test_dataloader, epochs=1)
torch.cuda.empty_cache()
trainer.load_checkpoint()
trainer.train_model()

#%%
###### test ######
# test the distribution of noise distribution
for xt, xt_, _, _  in test_dataloader:
    xt = xt.to(device); xt_ = xt_.to(device)
    yt_mean, yt_logvar, yt_estimate, yt_estimate_, y_mean_prior, y_logvar_prior, recon_xt, recon_xt_ = vae_model.forward(xt, xt_)
    # yt_truth_np_ = yt_.cpu().detach().numpy()
    # yt_estimate_np_ = yt_estimate_.cpu().detach().numpy()
    # print("yt_truth is:", yt_truth_np_.shape) # (3200, 2)
    # print("yt_estimated is:", yt_estimate_np_.shape) # (3200, 2)
    break

#%%
'''
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
'''

#%%
'''
i = 5
for a, b, c, d in test_dataloader:
    # visualization
    aa = a[i,:,:,:].permute(1,2,0).numpy()
    plt.figure()
    plt.imshow(aa)

    bb = b[i,:,:,:].permute(1,2,0).numpy()
    plt.figure()
    plt.imshow(bb)

    # label
    cc = c[i,:]
    print(cc.numpy())
    dd = d[i,:]
    print(dd.numpy())

    break

and we will have:
    list i including 4 components
    ~ torch.Size([BS, 3, 64, 64])
    ~ torch.Size([BS, 3, 64, 64])
    ~ torch.Size([BS, 8])
    ~ torch.Size([BS, 8])
'''





