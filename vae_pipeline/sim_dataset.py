"""
Generate simulated data in both sparse nonlinear and post nonlinear case

y_t = f1(A*y_(t-1)) + e_t, where f1 is nonlinear function, et ~ Gaussian distribution
or 
y_t = f2(f1(y_(t-1)) + e_t), where f1 and f2 are nonlinear function, et ~ Gaussian/NonGaussian distribution
x_t = g(y_t)

"""
import os
import glob
import torch
import random
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from sklearn import preprocessing

VALIDATION_RATIO = 0.2
standard_scaler = preprocessing.StandardScaler()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lags = 1
chunks = 1000
latent_size = 2
batch_size = 64
length = 80 + lags

nonlinearity = nn.LeakyReLU(0.2)
mixing_func = nn.Sequential(
	nn.Linear(latent_size, latent_size),
	nonlinearity,
	nn.Linear(latent_size, latent_size),
	nonlinearity,
	nn.Linear(latent_size, latent_size),
	nonlinearity
	)

def leaky_ReLU_1d(d, negSlope):
	if d > 0:
		return d
	else:
		return d * negSlope

leaky1d = np.vectorize(leaky_ReLU_1d)

def leaky_ReLU(D, negSlope):
	"""
	implementation of leaky ReLU activation function
	"""
	assert negSlope > 0  # must be positive
	return leaky1d(D, negSlope)

'''
# test latent distribution
plt.figure(figsize=(10,5))
array = np.asarray(latent_data)
z_np = array.reshape(-1,latent_size)
plt.hist(z_np[:,0])
plt.hist(z_np[:,1])
'''

def post_nonlinear_Gaussian():
	for chunk_idx in range(chunks):
		batch_data = []
		latent_data = []
		loc =  np.array([0, 1])
		scale = np.array([5, 1])
		# random initial y0 which follows normal distribution (BS, latent_dim)
		latents = np.random.normal(loc, scale, (batch_size, latent_size))

		for t in range(length):
			latent_data.append(latents)
			# x_t = g(y_t)
			mixedDat = mixing_func(torch.from_numpy(latents).float()).detach().numpy()
			batch_data.append(mixedDat)

			# y_t
			midDat = np.copy(latents) 
			# y_(t+1) = f2(f1(y_t) + e_t)
			midDat = leaky_ReLU(midDat, 0.2) + 0.3*np.random.normal(0, 2, (batch_size, latent_size)) 
			midDat = mixing_func(torch.from_numpy(midDat).float()).detach().numpy()
			latents = midDat

		batch_data = np.stack(batch_data, axis=1)
		latent_data = np.stack(latent_data, axis=1)
		np.savez("./dataset/post_nonlinear_Gaussian/%d"%chunk_idx, yt=latent_data[:,:-1,:], yt_=latent_data[:,1:,:], xt=batch_data[:,:-1,:], xt_=batch_data[:,1:,:])

def post_nonlinear_nonGaussian():
	for chunk_idx in range(chunks):
		batch_data = []
		latent_data = []
		loc =  np.array([0, 1])
		scale = np.array([5, 1])
		latents = np.random.normal(loc, scale, (batch_size, latent_size))

		for t in range(length):
			latent_data.append(latents)
			# x_t = g(y_t)
			mixedDat = mixing_func(torch.from_numpy(latents).float()).detach().numpy()
			batch_data.append(mixedDat)

			# y_t
			midDat = np.copy(latents) 
			# y_(t+1) = f2(f1(y_t) + e_t)
			midDat = leaky_ReLU(midDat, 0.2) + 0.3*np.random.laplace(0, 2, (batch_size, latent_size)) 
			midDat = mixing_func(torch.from_numpy(midDat).float()).detach().numpy()
			latents = midDat

		batch_data = np.stack(batch_data, axis=1)
		latent_data = np.stack(latent_data, axis=1)
		np.savez("./dataset/post_nonlinear_nonGaussian/%d"%chunk_idx, yt=latent_data[:,:-1,:], yt_=latent_data[:,1:,:], xt=batch_data[:,:-1,:], xt_=batch_data[:,1:,:])

def sparse_nonlinear():
	l1 = np.random.rand(2)
	mask = np.diag(l1)

	for chunk_idx in range(chunks):
		batch_data = []
		latent_data = []
		loc =  np.array([0, 1])
		scale = np.array([5, 1])
		latents = np.random.normal(loc, scale, (batch_size, latent_size))

		for t in range(length):
			latent_data.append(latents)
			# x_t = g(y_t)
			mixedDat = mixing_func(torch.from_numpy(latents).float()).detach().numpy()
			batch_data.append(mixedDat)

			# y_t
			midDat = np.copy(latents) 
			# y_t = f1(A \times y_t) + e_t
			midDat = np.matmul(mask, midDat.T).T
			midDat = mixing_func(torch.from_numpy(midDat).float()).detach().numpy()
			midDat = midDat + 0.3*np.random.normal(0, 2, (batch_size, latent_size)) 
			latents = midDat

		batch_data = np.stack(batch_data, axis=1)
		latent_data = np.stack(latent_data, axis=1)
		np.savez("./dataset/sparse_nonlinear/%d"%chunk_idx, yt=latent_data[:,:-1,:], yt_=latent_data[:,1:,:], xt=batch_data[:,:-1,:], xt_=batch_data[:,1:,:])

if __name__ == "__main__":
	post_nonlinear_Gaussian()
	datum_names = glob.glob(os.path.join("./dataset/post_nonlinear_Gaussian/", "*.npz"))
	n_samples = len(datum_names)
	# Shuffle samples
	random.shuffle(datum_names)
	n_train_samples = int((1-VALIDATION_RATIO)*n_samples)
	# Write training/val sample names to config files
	with open(os.path.join("./dataset/post_nonlinear_Gaussian/", "train.txt"), "w") as f:
		for datum_name in datum_names[:n_train_samples]:
			f.write('%s\n' % datum_name)
	with open(os.path.join("./dataset/post_nonlinear_Gaussian/", "val.txt"), "w") as f:
		for datum_name in datum_names[n_train_samples:]:
			f.write('%s\n' % datum_name)

	post_nonlinear_nonGaussian()
	datum_names = glob.glob(os.path.join("./dataset/post_nonlinear_nonGaussian/", "*.npz"))
	n_samples = len(datum_names)
	random.shuffle(datum_names)
	n_train_samples = int((1-VALIDATION_RATIO)*n_samples)
	with open(os.path.join("./dataset/post_nonlinear_nonGaussian/", "train.txt"), "w") as f:
		for datum_name in datum_names[:n_train_samples]:
			f.write('%s\n' % datum_name)
	with open(os.path.join("./dataset/post_nonlinear_nonGaussian/", "val.txt"), "w") as f:
		for datum_name in datum_names[n_train_samples:]:
			f.write('%s\n' % datum_name)

	sparse_nonlinear()
	datum_names = glob.glob(os.path.join("./dataset/sparse_nonlinear/", "*.npz"))
	n_samples = len(datum_names)
	random.shuffle(datum_names)
	n_train_samples = int((1-VALIDATION_RATIO)*n_samples)
	with open(os.path.join("./dataset/sparse_nonlinear/", "train.txt"), "w") as f:
		for datum_name in datum_names[:n_train_samples]:
			f.write('%s\n' % datum_name)
	with open(os.path.join("./dataset/sparse_nonlinear/", "val.txt"), "w") as f:
		for datum_name in datum_names[n_train_samples:]:
			f.write('%s\n' % datum_name)
