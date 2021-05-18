"""
Generate simulated data in both post nonlinear case

y_t = f2(f1(y_(t-1)) + e_t), where f1 and f2 are nonlinear function, et ~ Gaussian distribution
x_t = g(y_t)

three nonlinear cases are considered:
    sigmoid: invertiable
    leaky_relu: invertiable
    self_defined: invertiable and follows gaussian distribution

"""
import os
import glob
import torch
import random
import numpy as np
from sklearn import preprocessing

VALIDATION_RATIO = 0.2
# DIR = "post_nonlinear_data/f2_sigmoid/"  # in sigmoid case
DIR = "post_nonlinear_data/f2_leaky_relu/"  # in leaky_relu case
# DIR = "post_nonlinear_data/f2_self_defined/" # in self_defined case
standard_scaler = preprocessing.StandardScaler()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def self_defined(x):
	"""
	y = -x, if x \in [-1,1]
	y = x, else
	"""
	y = np.zeros((x.shape[0],x.shape[1]))
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			if x[i,j] > -1 and x[i,j] < 1:
				y[i,j] = -x[i,j]
			else:
				y[i,j] = x[i,j]
	return y

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

def sigmoidAct(x):
	"""
	one dimensional application of sigmoid activation function
	"""
	x =  1. / (1 + np.exp(-1 * x))
	return x

def main():
	lags = 1
	chunks = 20
	latent_size = 2
	batch_size = 64
	length = 500 + lags
	nonlinear_type = 2 # 1 for sigmoid; 2 for leaky_relu; 3 for self_defined

	for chunk_idx in range(chunks):
		batch_data = []
		latent_data = []
		loc =  np.array([0, 1])
		scale = np.array([5, 1])
		# random samples follow normal distribution
		latents = np.random.normal(loc, scale, (batch_size, latent_size))

		if nonlinear_type == 1:
			# sigmoid case
			for t in range(length):
				latent_data.append(latents)
				mixedDat = sigmoidAct(latents) # x_t = g(y_t)
				batch_data.append(mixedDat) 
				midDat = np.copy(latents) # y_(t-1)
				midDat = sigmoidAct(midDat) + 0.05 * np.random.normal(0, 1, (batch_size, latent_size)) # y_t = f1(y_(t-1)) + e_t
				midDat = sigmoidAct(midDat) # y_t = f2()
				latents = midDat
		elif nonlinear_type == 2: 
			# leaky_relu case
			for t in range(length):
				latent_data.append(latents)
				mixedDat = sigmoidAct(latents) # x_t = g(y_t)
				batch_data.append(mixedDat) 
				midDat = np.copy(latents) # y_(t-1)
				midDat = leaky_ReLU(midDat, 0.2) + 0.05 * np.random.normal(0, 1, (batch_size, latent_size)) # y_t = f1(y_(t-1)) + e_t
				midDat = leaky_ReLU(midDat, 0.2) # y_t = f2()
				latents = midDat
		else:
			# self_defined case
			for t in range(length):
				latent_data.append(latents)
				mixedDat = sigmoidAct(latents) # x_t = g(y_t)
				batch_data.append(mixedDat) 
				midDat = np.copy(latents) # y_(t-1)
				midDat = self_defined(midDat) + 0.2 * np.random.normal(0, 1, (batch_size, latent_size)) # y_t = f1(y_(t-1)) + e_t
				midDat = self_defined(midDat) # y_t = f2()
				latents = midDat
		'''
		# test latent distribution
		plt.figure(figsize=(10,5))
		array = np.asarray(latent_data)
		z_np = array.reshape(-1,latent_size)
		plt.hist(z_np[:,0])
		plt.hist(z_np[:,1])
		'''
		batch_data = np.stack(batch_data, axis=1) 
		latent_data = np.stack(latent_data, axis=1)

		if nonlinear_type == 1:
			np.savez("./post_nonlinear_data/f2_sigmoid/%d"%chunk_idx, yt=latent_data[:,:-1,:], yt_=latent_data[:,1:,:], xt=batch_data[:,:-1,:], xt_=batch_data[:,1:,:])
		elif nonlinear_type == 2: 
			np.savez("./post_nonlinear_data/f2_leaky_relu/%d"%chunk_idx, yt=latent_data[:,:-1,:], yt_=latent_data[:,1:,:], xt=batch_data[:,:-1,:], xt_=batch_data[:,1:,:])
		else:
			np.savez("./post_nonlinear_data/f2_self_defined/%d"%chunk_idx, yt=latent_data[:,:-1,:], yt_=latent_data[:,1:,:], xt=batch_data[:,:-1,:], xt_=batch_data[:,1:,:])

if __name__ == "__main__":
	main()
	datum_names = glob.glob(os.path.join(DIR, "*.npz"))
	n_samples = len(datum_names)
	# Shuffle samples
	random.shuffle(datum_names)
	n_train_samples = int((1-VALIDATION_RATIO)*n_samples)
	# Write training/val sample names to config files
	with open(os.path.join(DIR, "train.txt"), "w") as f:
		for datum_name in datum_names[:n_train_samples]:
			f.write('%s\n' % datum_name)
	with open(os.path.join(DIR, "val.txt"), "w") as f:
		for datum_name in datum_names[n_train_samples:]:
			f.write('%s\n' % datum_name)