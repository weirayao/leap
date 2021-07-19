import os
import glob
import tqdm
import torch
import scipy
import random
import ipdb as pdb
import numpy as np
from torch import nn
from collections import deque
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import ortho_group
from sklearn.preprocessing import scale

noise_scale = 0.1
VALIDATION_RATIO = 0.2
root_dir = '/home/yuewen/projects/data/'
standard_scaler = preprocessing.StandardScaler()

def leaky_ReLU_1d(d, negSlope):
    if d > 0:
        return d
    else:
        return d * negSlope

leaky1d = np.vectorize(leaky_ReLU_1d)

def leaky_ReLU(D, negSlope):
    assert negSlope > 0
    return leaky1d(D, negSlope)

def sigmoidAct(x):
    return 1. / (1 + np.exp(-1 * x))

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
    lags = 2
    Nlayer = 3
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 1000000
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "linear_nongaussian")
    os.makedirs(path, exist_ok=True)

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
    y_t = torch.distributions.laplace.Laplace(0,noise_scale).rsample((batch_size, latent_size)).numpy()
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

def linear_nonGaussian_ts():
    lags = 2
    Nlayer = 3
    length = 10
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 50000
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "linear_nongaussian_ts")
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
        mixingList.append(A)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    yt = []; xt = []
    for i in range(lags):
        yt.append(y_l[:,i,:])
    mixedDat = np.copy(y_l)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])
        
    # Mixing function
    for i in range(length):
        # Transition function
        y_t = torch.distributions.laplace.Laplace(0,noise_scale).rsample((batch_size, latent_size)).numpy()
        # y_t = (y_t - np.mean(y_t, axis=0 ,keepdims=True)) / np.std(y_t, axis=0 ,keepdims=True)
        for l in range(lags):
            y_t += np.dot(y_l[:,l,:], transitions[l])
        yt.append(y_t)
        # Mixing function
        mixedDat = np.copy(y_t)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_t = np.copy(mixedDat)
        xt.append(x_t)
        y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)

    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)

def nonlinear_Gaussian_ts():
    lags = 2
    Nlayer = 3
    length = 10
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 50000
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "nonlinear_gaussian_ts")
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 15)  # only accept those below 25% percentile
    for l in range(lags-1):
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

    yt = []; xt = []
    for i in range(lags):
        yt.append(y_l[:,i,:])
    mixedDat = np.copy(y_l)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])

    # Mixing function
    for i in range(length):
        # Transition function
        y_t = torch.distributions.normal.Normal(0,noise_scale).rsample((batch_size, latent_size)).numpy()
        # y_t = (y_t - np.mean(y_t, axis=0 ,keepdims=True)) / np.std(y_t, axis=0 ,keepdims=True)
        mixedDat = y_l[:,0,:] + y_l[:,1,:]
        for l in range(lags-1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            # mixedDat = sigmoidAct(mixedDat)
            mixedDat = np.dot(mixedDat, transitions[l])
        y_t = mixedDat + y_t
        yt.append(y_t)

        # Mixing function
        mixedDat = np.copy(y_t)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_t = np.copy(mixedDat)
        xt.append(x_t)
        y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)
    
    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)

if __name__ == "__main__":
    # linear_nonGaussian()
    linear_nonGaussian_ts()
    nonlinear_Gaussian_ts()
