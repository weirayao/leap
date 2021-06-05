"""
Generate simulated data in both sparse nonlinear and post nonlinear case

y_t = f1(A*y_(t-1)) + e_t, where f1 is nonlinear function, et ~ Gaussian distribution
or 
y_t = f2(f1(y_(t-1)) + e_t), where f1 and f2 are nonlinear function, et ~ Gaussian/NonGaussian distribution
x_t = g(y_t)

"""
import os
import glob
import tqdm
import torch
import scipy
import random
import numpy as np
from torch import nn
from collections import deque
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import ortho_group
from sklearn.preprocessing import scale

import ipdb as pdb
VALIDATION_RATIO = 0.2
standard_scaler = preprocessing.StandardScaler()

lags = 2
chunks = 500
latent_size = 4
input_size = 4
batch_size = 64
length = 160 + lags

root_dir = '/home/cmu_wyao/projects/data/'

nonlinearity = nn.LeakyReLU(0.2)
mixing_func = nn.Sequential(
    nn.Linear(latent_size, latent_size),
    nonlinearity,
    nn.Linear(latent_size, latent_size),
    nonlinearity,
    nn.Linear(latent_size, input_size),
    )

trans_func = nn.Sequential(
    nn.Linear(latent_size, latent_size),
    nonlinearity,
    nn.Linear(latent_size, latent_size),
    nonlinearity,
    nn.Linear(latent_size, latent_size),
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
def generateUniformMat(Ncomp, condT):
    """
    generate a random matrix by sampling each element uniformly at random
    check condition number versus a condition threshold
    """
    A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
    for i in range(Ncomp):
        A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    while np.linalg.cond(A) > condT:
        # generate a new A matrix!
        A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
        for i in range(Ncomp):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    return A

def linear_nonGaussian():
    batch_size = 1000000
    Nlayer = 3
    negSlope = 0.2
    Niter4condThresh = 1e4
    path = os.path.join(root_dir, "linear_nongaussian")
    os.makedirs(path, exist_ok=True)
    transitions = [ ]
    condList = []
    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))
    condThresh = np.percentile(condList, 15)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    # Mixing function
    mixedDat = np.copy(y_l)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)
    # Transition function
    y_t = torch.distributions.laplace.Laplace(0,0.005).rsample((batch_size, latent_size)).numpy()
    # y_t = (y_t - np.mean(y_t, axis=0 ,keepdims=True)) / np.std(y_t, axis=0 ,keepdims=True)
    for l in range(lags):
        y_t += np.dot(y_l[:,l,:], transitions[l])
    # Mixing function
    mixedDat = np.copy(y_t)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_t = np.copy(mixedDat)

    np.savez(os.path.join(path, "data"), 
             yt = y_l, 
             yt_ = y_t, 
             xt = x_l, 
             xt_= x_t)

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)

def linear_nonGaussian_deprecated():
    # Super-Gaussian is exp(Z) of standard normals
    # Sub-Gaussian is Laplace distribution
    path = os.path.join(root_dir, "linear_nongaussian")
    os.makedirs(path, exist_ok=True)
    transitions = [ ]
    scale = 2
    for l in range(lags):
        B = ((torch.rand(latent_size, latent_size) - 0.5)/scale)
        scale = scale * 2
        transitions.append(B)
    transitions.reverse()
    xt_ = [ ]
    yt_ = [ ]
    xt = [ ]
    yt = [ ]
    for chunk_idx in tqdm.tqdm(range(chunks)):
        curr_batch_data = [ ]
        past_batch_data = [ ]
        curr_latent_data = [ ]
        past_latent_data = [ ]
        # Initialize past latents
        y_l = torch.randn(batch_size, lags, latent_size)
        x_l = mixing_func(y_l)
        for t in range(length):
            past_latent_data.append(y_l.detach().cpu().numpy())
            past_batch_data.append(x_l.detach().cpu().numpy())
            # Sample current noise y_t = Laplacian(0, 1)
            y_t = torch.distributions.laplace.Laplace(0,1).rsample((batch_size, latent_size))
            for l in range(lags):
                y_t += torch.mm(y_l[:,l,:], transitions[l])
            x_t = mixing_func(y_t)
            curr_latent_data.append(y_t.detach().cpu().numpy())
            curr_batch_data.append(x_t.detach().cpu().numpy())
            # Update past latents
            y_l = torch.cat((y_l[:,1:], y_t.unsqueeze(1)), dim=1)
            x_l = mixing_func(y_l)

        # batch_data = [BS, length, input_size]
        curr_batch_data = np.concatenate(curr_batch_data[80:], axis=0)
        past_batch_data = np.concatenate(past_batch_data[80:], axis=0)
        curr_latent_data = np.concatenate(curr_latent_data[80:], axis=0)
        past_latent_data = np.concatenate(past_latent_data[80:], axis=0)
        xt_.append(curr_batch_data)
        yt_.append(curr_latent_data)
        xt.append(past_batch_data)
        yt.append(past_latent_data)
    xt_ = np.concatenate(xt_, axis=0)
    yt_ = np.concatenate(yt_, axis=0)
    xt = np.concatenate(xt, axis=0)
    yt = np.concatenate(yt, axis=0)
    np.savez(os.path.join(path, "data"), 
             yt = yt, yt_ = yt_, xt = xt, xt_= xt_)
    for l in range(lags):
        B = transitions[l].detach().cpu().numpy()
        np.save(os.path.join(path, "W%d"%(lags-l)), B)



# def linear_nonGaussian():
# 	# Super-Gaussian is exp(Z) of standard normals
# 	# Sub-Gaussian is Laplace distribution
# 	path = os.path.join(root_dir, "linear_nongaussian")
# 	os.makedirs(path, exist_ok=True)
# 	transitions = [ ]
# 	scale = 2
# 	for l in range(lags):
# 		B = ((torch.rand(latent_size, latent_size) - 0.5)/scale)
# 		scale = scale * 2
# 		transitions.append(B)
# 	transitions.reverse()
# 	xt_ = [ ]
# 	yt_ = [ ]
# 	xt = [ ]
# 	yt = [ ]
# 	for chunk_idx in tqdm.tqdm(range(chunks)):
# 		curr_batch_data = [ ]
# 		past_batch_data = [ ]
# 		curr_latent_data = [ ]
# 		past_latent_data = [ ]
# 		# Initialize past latents
# 		y_l = torch.rand(batch_size, lags, latent_size)
# 		x_l = mixing_func(y_l)
# 		for t in range(length):
# 			past_latent_data.append(y_l.detach().cpu().numpy())
# 			past_batch_data.append(x_l.detach().cpu().numpy())
# 			# Sample current noise y_t = [y_1, y_2]
# 			y_1 = torch.exp(torch.normal(0, 1, size=(batch_size, latent_size//2)))
# 			y_2 = torch.distributions.laplace.Laplace(0,1).rsample((batch_size, latent_size//2))
# 			y_t = torch.cat((y_1, y_2), dim=1)
# 			for l in range(lags):
# 				y_t += torch.mm(y_l[:,l,:], transitions[l])
# 			x_t = mixing_func(y_t)
# 			curr_latent_data.append(y_t.detach().cpu().numpy())
# 			curr_batch_data.append(x_t.detach().cpu().numpy())
# 			# Update past latents
# 			y_l = torch.cat((y_l[:,1:], y_t.unsqueeze(1)), dim=1)
# 			x_l = mixing_func(y_l)

# 		# batch_data = [BS, length, input_size]
# 		curr_batch_data = np.concatenate(curr_batch_data, axis=0)
# 		past_batch_data = np.concatenate(past_batch_data, axis=0)
# 		curr_latent_data = np.concatenate(curr_latent_data, axis=0)
# 		past_latent_data = np.concatenate(past_latent_data, axis=0)
# 		xt_.append(curr_batch_data)
# 		yt_.append(curr_latent_data)
# 		xt.append(past_batch_data)
# 		yt.append(past_latent_data)
# 	xt_ = np.concatenate(xt_, axis=0)
# 	yt_ = np.concatenate(yt_, axis=0)
# 	xt = np.concatenate(xt, axis=0)
# 	yt = np.concatenate(yt, axis=0)
# 	np.savez(os.path.join(path, "data"), 
# 			 yt = yt, yt_ = yt_, xt = xt, xt_= xt_)
# 	for l in range(lags):
# 		B = transitions[l].detach().cpu().numpy()
# 		np.save(os.path.join(path, "W%d"%(lags-l)), B)

def post_nonlinear_Gaussian():
    lags = 1
    Nlayer = 3
    negSlope = 0.2
    batch_size = 1000000
    path = os.path.join(root_dir, "post_nonlinear_gaussian")
    os.makedirs(path, exist_ok=True)

    # why
    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    # Mixing function
    mixedDat = np.copy(y_l)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)

    # Transition function
    y_t = torch.normal(0, 0.005, size=(batch_size, latent_size)).numpy()
    # y_t = (y_t - np.mean(y_t, axis=0 ,keepdims=True)) / np.std(y_t, axis=0 ,keepdims=True)
    for l in range(lags):
        y_t += leaky_ReLU(y_l[:,l,:], negSlope)
        y_t = np.dot(y_t, mixingList[l])
    y_t = leaky_ReLU(y_t, negSlope)
    # y_t = np.dot(y_t, mixingList[l])

    # Mixing function
    mixedDat = np.copy(y_t)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_t = np.copy(mixedDat)

    np.savez(os.path.join(path, "data"), 
            yt = y_l, # (64, 1, 4)
            yt_ = y_t, # (64, 4)
            xt = x_l, # (64, 1, 4)
            xt_= x_t) # (64, 4)

def post_nonlinear_nonGaussian():
    lags = 1
    Nlayer = 3
    negSlope = 0.2
    batch_size = 1000000
    path = os.path.join(root_dir, "post_nonlinear_nongaussian")
    os.makedirs(path, exist_ok=True)

    # why
    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    # Mixing function
    mixedDat = np.copy(y_l)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)

    # Transition function
    y_t = torch.distributions.laplace.Laplace(0,0.005).rsample((batch_size, latent_size)).numpy()
    # y_t = (y_t - np.mean(y_t, axis=0 ,keepdims=True)) / np.std(y_t, axis=0 ,keepdims=True)
    for l in range(lags):
        y_t += leaky_ReLU(y_l[:,l,:], negSlope)
        y_t = np.dot(y_t, mixingList[l])
    y_t = leaky_ReLU(y_t, negSlope)
    # y_t = np.dot(y_t, mixingList[l])

    # Mixing function
    mixedDat = np.copy(y_t)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_t = np.copy(mixedDat)

    np.savez(os.path.join(path, "data"), 
            yt = y_l, # (64, 1, 4)
            yt_ = y_t, # (64, 4)
            xt = x_l, # (64, 1, 4)
            xt_= x_t) # (64, 4)

# def sparse_nonlinear():
# 	path = os.path.join(root_dir, "post_nonlinear_nongaussian")
# 	os.makedirs(path, exist_ok=True)
# 	l1 = np.random.rand(2)
# 	mask = np.diag(l1)

# 	for chunk_idx in range(chunks):
# 		batch_data = []
# 		latent_data = []
# 		loc =  np.array([0, 1])
# 		scale = np.array([5, 1])
# 		latents = np.random.normal(loc, scale, (batch_size, latent_size))

# 		for t in range(length):
# 			latent_data.append(latents)
# 			# x_t = g(y_t)
# 			mixedDat = mixing_func(torch.from_numpy(latents).float()).detach().numpy()
# 			batch_data.append(mixedDat)

# 			# y_t
# 			midDat = np.copy(latents) 
# 			# y_t = f1(A \times y_t) + e_t
# 			midDat = np.matmul(mask, midDat.T).T
# 			midDat = mixing_func(torch.from_numpy(midDat).float()).detach().numpy()
# 			midDat = midDat + 0.3*np.random.normal(0, 2, (batch_size, latent_size)) 
# 			latents = midDat

# 		batch_data = np.stack(batch_data, axis=1)
# 		latent_data = np.stack(latent_data, axis=1)
# 		np.savez(os.path.join(path,"%d"%chunk_idx), 
# 				 yt=latent_data[:,:-1,:], yt_=latent_data[:,1:,:], xt=batch_data[:,:-1,:], xt_=batch_data[:,1:,:])

if __name__ == "__main__":
    linear_nonGaussian()
    post_nonlinear_Gaussian()
    post_nonlinear_nonGaussian()

    # datum_names = glob.glob(os.path.join("./dataset/post_nonlinear_Gaussian/", "*.npz"))
    # n_samples = len(datum_names)
    # # Shuffle samples
    # random.shuffle(datum_names)
    # n_train_samples = int((1-VALIDATION_RATIO)*n_samples)
    # # Write training/val sample names to config files
    # with open(os.path.join("./dataset/post_nonlinear_Gaussian/", "train.txt"), "w") as f:
    # 	for datum_name in datum_names[:n_train_samples]:
    # 		f.write('%s\n' % datum_name)
    # with open(os.path.join("./dataset/post_nonlinear_Gaussian/", "val.txt"), "w") as f:
    # 	for datum_name in datum_names[n_train_samples:]:
    # 		f.write('%s\n' % datum_name)

    # post_nonlinear_nonGaussian()
    # datum_names = glob.glob(os.path.join("./dataset/post_nonlinear_nonGaussian/", "*.npz"))
    # n_samples = len(datum_names)
    # random.shuffle(datum_names)
    # n_train_samples = int((1-VALIDATION_RATIO)*n_samples)
    # with open(os.path.join("./dataset/post_nonlinear_nonGaussian/", "train.txt"), "w") as f:
    # 	for datum_name in datum_names[:n_train_samples]:
    # 		f.write('%s\n' % datum_name)
    # with open(os.path.join("./dataset/post_nonlinear_nonGaussian/", "val.txt"), "w") as f:
    # 	for datum_name in datum_names[n_train_samples:]:
    # 		f.write('%s\n' % datum_name)

    # sparse_nonlinear()
    # datum_names = glob.glob(os.path.join("./dataset/sparse_nonlinear/", "*.npz"))
    # n_samples = len(datum_names)
    # random.shuffle(datum_names)
    # n_train_samples = int((1-VALIDATION_RATIO)*n_samples)
    # with open(os.path.join("./dataset/sparse_nonlinear/", "train.txt"), "w") as f:
    # 	for datum_name in datum_names[:n_train_samples]:
    # 		f.write('%s\n' % datum_name)
    # with open(os.path.join("./dataset/sparse_nonlinear/", "val.txt"), "w") as f:
    # 	for datum_name in datum_names[n_train_samples:]:
    # 		f.write('%s\n' % datum_name)
