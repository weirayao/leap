# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:04:17 2021
reference code:
    https://github.com/AntixK/PyTorch-VAE/blob/8700d245a9735640dda458db4cf40708caf2e77f/models/vanilla_vae.py
    https://github.com/MelJan/PyDeep/blob/84c5d41942ab114bc8827ea2c10ea0a3d6dfdfd3/pydeep/misc/visualization.py#L692
@author: 
"""

# training the vae model
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn import preprocessing
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor

from utils import SimulationDataset
from spline import ComponentWiseSpline

length = 50
batch_size = 64
linear_type = True # False in nonlinear case
lb = length * batch_size
Tensor = TypeVar('torch.tensor')
M_true = np.array([[1, -1.5], [0, 1.2]])
standard_scaler = preprocessing.StandardScaler()
# min_max_scaler = preprocessing.MinMaxScaler()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###### VAE model ######
class TemporalVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=32, **kwargs):
        super(TemporalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.spline = ComponentWiseSpline(input_dim)
        # Build Encoder
        self.encoder = nn.Sequential(
                        nn.Linear(input_dim*2, 16), 
                        # concat [xt & xt_1]
                        nn.BatchNorm1d(16),
                        nn.LeakyReLU(),                                        
                        nn.Linear(16, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU()
                        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # Build Coupling
        self.f1 = nn.Sequential(
                    nn.Linear(input_dim, 16),
                    nn.BatchNorm1d(16),
                    nn.LeakyReLU(),
                    nn.Linear(16, 32),
                    nn.BatchNorm1d(32),
                    nn.LeakyReLU(),
                    nn.Linear(32, latent_dim),
                    nn.BatchNorm1d(latent_dim),
                    nn.LeakyReLU()
                    )
        self.M = nn.Linear(latent_dim, latent_dim, bias=False)
        # parameter M for the nonlinearity yt = g(M * yt_1, e_t)
        self.f2 = nn.Sequential(
                    nn.Linear(latent_dim, latent_dim),
                    nn.BatchNorm1d(latent_dim),
                    nn.LeakyReLU()
                    )

        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.Sequential(
                        nn.Linear(hidden_dim, 16),
                        nn.BatchNorm1d(16),
                        nn.LeakyReLU(),
                        nn.Linear(16, input_dim),
                        nn.BatchNorm1d(input_dim),
                        nn.LeakyReLU()
                        )

    def encode(self, xt, xt_1):
        """
        Encodes the input x_t and condition x_(t-1) by passing through the encoder network
        and returns the latent codes \mu and \log_var.
        : param input: (Tensor) Input tensor to encoder [BS x length, obs_dim]
        : param condition: (Tensor) Input tensor to encoder [BS x length, obs_dim]
        : return: (Tensor) List of latent codes
        """
        x = torch.cat((xt, xt_1), 1) 
        result = self.encoder(x) 
        # Split the result into mu and var components of the latent Gaussian distribution/noise term
        mu = self.fc_mu(result) 
        log_var = self.fc_var(result) 
        return mu, log_var

    def decode(self, yt):
        """
        Maps the given latent codes and coupling onto the observation space.
        : param z: (Tensor) [hidden_dim x latent_dim]
        : return: (Tensor) [BS x length x obs_dim]
        """
        result = self.decoder_input(yt) 
        result = self.decoder(result) 
        result = result.view(-1, 2)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        : param mu: (Tensor) Mean of the latent Gaussian 
        : param logvar: (Tensor) Standard deviation of the latent Gaussian 
        : return: (Tensor) 
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, xt, xt_1, **kwargs):
        """
        Encoder and coupling part.
        : yt_1 = h(x)
        : yt = g(M*yt_1) + z
        : return: recon_x, observed_x, z, mu, log_var, M
        """
        mu, log_var = self.encode(xt, xt_1) 
        z = self.reparameterize(mu, log_var) 
        if not linear_type:
        	z, _ = self.spline(z)
        yt_1 = self.f1(xt_1) 
        yt_ = self.M(yt_1) 
        if linear_type:
        	yt = yt_ + z 
    	else:
    		yt = self.f2(yt_) + z 
        
        return self.decode(yt), xt, z, mu, log_var, self.M.weight

    def generate(self, xt, xt_1, **kwargs):
        """
        Given an input x, returns the reconstructed \tilda x
        : param: xt, xt_1 (Tensor) 
        : return: \tilda x, latent z (Tensor) 
        """
        gen, _, z, _, _, _ = self.forward(xt, xt_1)
        return gen, z

#### vae loss function ####
def loss_func(recon, xt, mu, log_var, M):
    """
    Computes the VAE loss function.
    Add L1 loss for sparse constraint.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}

    """
    recons_loss = F.mse_loss(recon, xt)
    if linear_type:
    	kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))
    else:
    	pass
    loss_vae = recons_loss + kld_loss
    loss_filter = torch.mean(torch.abs(M))
    loss = loss_vae + loss_filter
    return loss, recons_loss, kld_loss, loss_filter

#### calculate amari distance ####
def calculate_amari_distance(matrix_one,matrix_two,version=1):
    """ Calculate the Amari distance between two input matrices.
    :param matrix_one: the first matrix
    :type matrix_one: numpy array
    :param matrix_two: the second matrix
    :type matrix_two: numpy array
    :param version: Variant to use.
    :type version: int
    :return: The amari distance between two input matrices.
    :rtype: float
    """
    if matrix_one.shape != matrix_two.shape:
        return "Two matrices must have the same shape."
    product_matrix = np.abs(np.dot(matrix_one,np.linalg.inv(matrix_two)))
    product_matrix_max_col = np.array(product_matrix.max(0))
    product_matrix_max_row = np.array(product_matrix.max(1))

    n = product_matrix.shape[0]

    if version != 1:
        """ Formula from Teh
        Here they refered to as "amari distance"
        The value is in [2*N-2N^2, 0].
        reference:
            Teh, Y. W.; Welling, M.; Osindero, S. & Hinton, G. E. Energy-based
            models for sparse overcomplete representations J MACH LEARN RES,
            2003, 4, 1235--1260
        """
        amari_distance = product_matrix / np.tile(product_matrix_max_col, (n, 1))
        amari_distance += product_matrix / np.tile(product_matrix_max_row, (n, 1)).T
        amari_distance = amari_distance.sum() - 2 * n * n
    else:
        """ Formula from ESLII
        Here they refered to as "amari error"
        The value is in [0, N-1].
        reference:
            Bach, F. R.; Jordan, M. I. Kernel Independent Component
            Analysis, J MACH LEARN RES, 2002, 3, 1--48
        """
        amari_distance = product_matrix / np.tile(product_matrix_max_col, (n, 1))
        amari_distance += product_matrix / np.tile(product_matrix_max_row, (n, 1)).T
        amari_distance = amari_distance.sum() / (2 * n) - 1
    return amari_distance

###### training model ######
val_meanloss_list = []
class Trainer(object):
    def __init__(self, model, train_dataloader, test_dataloader, learning_rate=5e-3, epochs=None, ckpoint='./ckpoint/vae_model.pth'):
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
        val_losses = []
        for batch_sample in self.test_dataloader:
            xt_1 = batch_sample["xt_1"].reshape((lb,-1))
            xt = batch_sample["xt"].reshape((lb,-1))
            xt_1 = torch.from_numpy(standard_scaler.fit_transform(xt_1)).to(device).float()
            xt = torch.from_numpy(standard_scaler.fit_transform(xt)).to(device).float()
            recon_x, x, _, mu, logvar, M = self.model.forward(xt, xt_1)
            loss, recon, kld, matrix = loss_func(recon_x, x, mu, logvar, M)
            val_losses.append(loss.item())
        val_meanloss = np.mean(val_losses)
        val_meanloss_list.append(val_meanloss)
        self.model.train()
        retun val_meanloss_list

    def train_model(self):
        self.model.train()
        meanloss_list = []; distance_list = []
        for epoch in trange(self.start_epoch, self.epochs):
            losses = []; recons = []; klds = []; matrixs = []
            print("Running Epoch : {}".format(epoch+1))
            for batch_sample in self.train_dataloader:
                xt_1 = batch_sample["xt_1"].reshape((lb,-1))
                xt = batch_sample["xt"].reshape((lb,-1))
                xt_1 = torch.from_numpy(standard_scaler.fit_transform(xt_1)).to(device).float()
                xt = torch.from_numpy(standard_scaler.fit_transform(xt)).to(device).float()
                self.optimizer.zero_grad()
                recon_x, x, _, mu, logvar, M = self.model.forward(xt, xt_1)
                loss, recon, kld, matrix = loss_func(recon_x, x, mu, logvar, M)
                loss.backward()
                losses.append(loss.item()); recons.append(recon.item())
                klds.append(kld.item()); matrixs.append(matrix.item())
                self.optimizer.step()
            meanloss = np.mean(losses); meanrecon = np.mean(recons)
            meankld = np.mean(klds); meanmatrix = np.mean(matrixs)
            print("Epoch {}: average loss: {}, recon loss: {}, kld loss: {}, matrix loss: {}".format(epoch+1, meanloss, meanrecon, meankld, meanmatrix))
            distance = calculate_amari_distance(M_true, M.detach().cpu().numpy())
            meanloss_list.append(meanloss)
            distance_list.append(distance)
            self.save_checkpoint(epoch)
            val_meanloss_list = self.val_model()
        np.save("amari.npy", np.asarray(distance_list))
        np.save("training_loss.npy", np.asarray(meanloss_list))
        np.save("testing_loss.npy", np.asarray(val_meanloss_list))

###### training realization ######
test_dataset = SimulationDataset("val")
train_dataset = SimulationDataset("train")
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
vae_model = TemporalVAE(input_dim=2, latent_dim=2)
trainer = Trainer(vae_model, train_dataloader, test_dataloader, epochs=100)
torch.cuda.empty_cache()
trainer.load_checkpoint()
trainer.train_model()

###### test ######
# test the distribution of noise distribution
for batch_sample in test_dataloader:
    xt_1 = batch_sample["xt_1"].reshape((lb,-1))
    xt = batch_sample["xt"].reshape((lb,-1))
    xt_1 = torch.from_numpy(standard_scaler.fit_transform(xt_1)).to(device).float()
    xt = torch.from_numpy(standard_scaler.fit_transform(xt)).to(device).float()
    recon_x, z = vae_model.generate(xt, xt_1)
    # print("original x is:", xt.cpu().detach().numpy())
    # print("recon x is:", recon_x.cpu().detach().numpy())
    z_np = z.cpu().detach().numpy()
    break

print("distance between xt is: ", F.mse_loss(recon_x, xt))
plt.figure(figsize=(10,5))
plt.subplot(221)
plt.hist(z_np[:,0])
plt.subplot(222)
plt.hist(z_np[:,1])  

# test the amari_distancc(M, M_)
# & plot the loss curve of train_dataloader/test_dataloader
amari_distancc = np.load("amari.npy")
test_loss_curve = np.load("testing_loss.npy")
train_loss_curve = np.load("training_loss.npy")
plt.figure(figsize=(10,5))
plt.subplot(131)
plt.plot(amari_distancc)
plt.title("amari_distancc")
plt.subplot(132)
plt.plot(test_loss_curve)
plt.title("test_loss_curve")
plt.subplot(133)
plt.plot(train_loss_curve)
plt.title("train_loss_curve")