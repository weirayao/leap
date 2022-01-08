import os
import glob
import tqdm
import torch
import scipy
import random
import ipdb as pdb
import numpy as np
from torch import nn
from torch.nn import init
from collections import deque
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import ortho_group
from sklearn.preprocessing import scale
from leap.tools.utils import create_sparse_transitions, controlable_sparse_transitions

VALIDATION_RATIO = 0.2
root_dir = '/srv/data/ltcl/data'
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

def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()

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
    length = 4
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    noise_scale = 0.1
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
    length = 4
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    noise_scale = 0.1
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

    f2 = nn.LeakyReLU(0.2) # (1)3

    # Mixing function
    for i in range(length):
        # Transition function
        y_t = torch.distributions.normal.Normal(0, noise_scale).rsample((batch_size, latent_size)).numpy()
        # y_t = (y_t - np.mean(y_t, axis=0 ,keepdims=True)) / np.std(y_t, axis=0 ,keepdims=True)
        for l in range(lags):
            y_t += np.tanh(np.dot(y_l[:,l,:], transitions[l]))
        y_t = leaky_ReLU(y_t, negSlope)
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

def nonlinear_Gaussian_ts_deprecated():
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

    f1 = nn.Sequential(nn.Linear(2*latent_size, latent_size), nn.LeakyReLU(0.2))
    f2 = nn.Sequential(nn.Linear(latent_size, latent_size), nn.LeakyReLU(0.2))
    # Mixing function
    for i in range(length):
        # Transition function
        y_t = torch.distributions.normal.Normal(0,noise_scale).rsample((batch_size, latent_size))
        # y_t = (y_t - np.mean(y_t, axis=0 ,keepdims=True)) / np.std(y_t, axis=0 ,keepdims=True)
        # pdb.set_trace()
        '''
        y_l1 = torch.from_numpy(np.dot(y_l[:,0,:], transitions[0]))
        y_l2 = torch.from_numpy(np.dot(y_l[:,1,:], transitions[1]))
        mixedDat = torch.cat([y_l1, y_l2], dim=1)
        mixedDat = f1(mixedDat.float()).detach().numpy()
        '''
        mixedDat = torch.from_numpy(y_l)
        mixedDat = torch.cat([mixedDat[:,0,:], mixedDat[:,1,:]], dim=1)
        mixedDat = torch.add(f1(mixedDat.float()), y_t)
        '''
        mixedDat = y_l[:,0,:] + y_l[:,1,:]
        for l in range(lags-1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            # mixedDat = sigmoidAct(mixedDat)
            mixedDat = np.dot(mixedDat, transitions[l])
        '''
        # y_t = leaky_ReLU(mixedDat + y_t, negSlope)
        y_t = f2(mixedDat).detach().numpy() # PNL
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

def nonlinear_Gaussian_ts_deprecated():
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

    f1 = nn.Sequential(nn.Linear(2*latent_size, latent_size), nn.LeakyReLU(0.2))
    # Mixing function
    for i in range(length):
        # Transition function
        y_t = torch.distributions.normal.Normal(0,noise_scale).rsample((batch_size, latent_size)).numpy()
        # y_t = (y_t - np.mean(y_t, axis=0 ,keepdims=True)) / np.std(y_t, axis=0 ,keepdims=True)
        # pdb.set_trace()
        
        y_l1 = torch.from_numpy(np.dot(y_l[:,0,:], transitions[0]))
        y_l2 = torch.from_numpy(np.dot(y_l[:,1,:], transitions[1]))
        mixedDat = torch.cat([y_l1, y_l2], dim=1)
        mixedDat = f1(mixedDat.float()).detach().numpy()
        '''
        mixedDat = torch.from_numpy(y_l)
        mixedDat = torch.cat([mixedDat[:,0,:], mixedDat[:,1,:]], dim=1)
        mixedDat = f1(mixedDat.float()).detach().numpy()
        '''
        '''
        mixedDat = y_l[:,0,:] + y_l[:,1,:]
        for l in range(lags-1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            # mixedDat = sigmoidAct(mixedDat)
            mixedDat = np.dot(mixedDat, transitions[l])
        '''

        y_t = leaky_ReLU(mixedDat + y_t, negSlope)
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

def nonlinear_nonGaussian_ts():
    lags = 2
    Nlayer = 3
    length = 4
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 50000
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "nonlinear_nongaussian_ts")
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

    # f1 = nn.Sequential(nn.Linear(2*latent_size, latent_size),
    #                    nn.LeakyReLU(0.2),
    #                    nn.Linear(latent_size, latent_size),
    #                    nn.LeakyReLU(0.2),
    #                    nn.Linear(latent_size, latent_size)) 
    # # f1.apply(weigth_init)
    f2 = nn.LeakyReLU(0.2) # (1)3

    # # Mixing function
    # for i in range(length):
    #     # Transition function
    #     y_t = torch.distributions.laplace.Laplace(0,noise_scale).rsample((batch_size, latent_size))
    #     # y_t = (y_t - np.mean(y_t, axis=0 ,keepdims=True)) / np.std(y_t, axis=0 ,keepdims=True)
    #     # pdb.set_trace()
    #     '''
    #     y_l1 = torch.from_numpy(np.dot(y_l[:,0,:], transitions[0]))
    #     y_l2 = torch.from_numpy(np.dot(y_l[:,1,:], transitions[1]))
    #     mixedDat = torch.cat([y_l1, y_l2], dim=1)
    #     mixedDat = f1(mixedDat.float()).detach().numpy()
    #     '''
    #     mixedDat = torch.from_numpy(y_l)
    #     # mixedDat = torch.cat([mixedDat[:,0,:], mixedDat[:,1,:]], dim=1)
    #     mixedDat = 2 * mixedDat[:,0,:] + mixedDat[:,1,:]
    #     mixedDat = torch.add(mixedDat.float(), y_t)
    #     '''
    #     mixedDat = y_l[:,0,:] + y_l[:,1,:]
    #     for l in range(lags-1):
    #         mixedDat = leaky_ReLU(mixedDat, negSlope)
    #         # mixedDat = sigmoidAct(mixedDat)
    #         mixedDat = np.dot(mixedDat, transitions[l])
    #     '''
    #     # y_t = leaky_ReLU(mixedDat + y_t, negSlope)
    #     # y_t = f2(mixedDat).detach().numpy() # PNL
    #     y_t = mixedDat.detach().numpy()
    #     yt.append(y_t)
    #     # Mixing function
    #     mixedDat = np.copy(y_t)
    #     for l in range(Nlayer - 1):
    #         mixedDat = leaky_ReLU(mixedDat, negSlope)
    #         mixedDat = np.dot(mixedDat, mixingList[l])
    #     x_t = np.copy(mixedDat)
    #     xt.append(x_t)
    #     # pdb.set_trace()
    #     y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    # Mixing function
    for i in range(length):
        # Transition function
        y_t = torch.distributions.laplace.Laplace(0,noise_scale).rsample((batch_size, latent_size)).numpy()
        # y_t = (y_t - np.mean(y_t, axis=0 ,keepdims=True)) / np.std(y_t, axis=0 ,keepdims=True)
        for l in range(lags):
            y_t += np.sin(np.dot(y_l[:,l,:], transitions[l]))
        y_t = leaky_ReLU(y_t, negSlope)
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

def nonlinear_ns():
    lags = 2
    Nlayer = 3
    length = 4
    Nclass = 3
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 50000
    Niter4condThresh = 1e4
    noise_scale = [0.05, 0.1, 0.15] # (v1)
    # noise_scale = [0.01, 0.1, 1]
    # noise_scale = [0.01, 0.05, 0.1] 

    path = os.path.join(root_dir, "nonlinear_ns")
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

    yt = []; xt = []; ct = []
    yt_ns = []; xt_ns = []; ct_ns = []

    # Mixing function
    for j in range(Nclass):
        ct.append(j * np.ones(batch_size))
        y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
        y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
        
        # Initialize the dataset
        for i in range(lags):
            yt.append(y_l[:,i,:])
        mixedDat = np.copy(y_l)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_l = np.copy(mixedDat)
        for i in range(lags):
            xt.append(x_l[:,i,:])
            
        # Generate time series dataset
        for i in range(length):
            # Transition function
            y_t = torch.distributions.laplace.Laplace(0,noise_scale[j]).rsample((batch_size, latent_size)).numpy()
            for l in range(lags):
                y_t += np.tanh(np.dot(y_l[:,l,:], transitions[l]))
            y_t = leaky_ReLU(y_t, negSlope)
            yt.append(y_t)

            # Mixing function
            mixedDat = np.copy(y_t)
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingList[l])
            x_t = np.copy(mixedDat)
            xt.append(x_t)

            y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]
        
        yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); ct = np.array(ct).transpose(1,0)
        yt_ns.append(yt); xt_ns.append(xt); ct_ns.append(ct)
        yt = []; xt = []; ct = []

    yt_ns = np.vstack(yt_ns)
    xt_ns = np.vstack(xt_ns)
    ct_ns = np.vstack(ct_ns)

    np.savez(os.path.join(path, "data"), 
            yt = yt_ns, 
            xt = xt_ns,
            ct = ct_ns)

def nonlinear_gau_ns():
    lags = 2
    Nlayer = 3
    length = 4
    Nclass = 3
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 50000
    Niter4condThresh = 1e4
    noise_scale = [0.05, 0.1, 0.15] # (v1)
    # noise_scale = [0.01, 0.1, 1]
    # noise_scale = [0.01, 0.05, 0.1] 

    path = os.path.join(root_dir, "nonlinear_gau_ns")
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

    yt = []; xt = []; ct = []
    yt_ns = []; xt_ns = []; ct_ns = []

    # Mixing function
    for j in range(Nclass):
        ct.append(j * np.ones(batch_size))
        y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
        y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
        
        # Initialize the dataset
        for i in range(lags):
            yt.append(y_l[:,i,:])
        mixedDat = np.copy(y_l)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_l = np.copy(mixedDat)
        for i in range(lags):
            xt.append(x_l[:,i,:])
            
        # Generate time series dataset
        for i in range(length):
            # Transition function
            y_t = torch.distributions.normal.Normal(0,noise_scale[j]).rsample((batch_size, latent_size)).numpy()
            for l in range(lags):
                y_t += np.sin(np.dot(y_l[:,l,:], transitions[l]))
            y_t = leaky_ReLU(y_t, negSlope)
            yt.append(y_t)

            # Mixing function
            mixedDat = np.copy(y_t)
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingList[l])
            x_t = np.copy(mixedDat)
            xt.append(x_t)

            y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]
        
        yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); ct = np.array(ct).transpose(1,0)
        yt_ns.append(yt); xt_ns.append(xt); ct_ns.append(ct)
        yt = []; xt = []; ct = []

    yt_ns = np.vstack(yt_ns)
    xt_ns = np.vstack(xt_ns)
    ct_ns = np.vstack(ct_ns)

    np.savez(os.path.join(path, "data"), 
            yt = yt_ns, 
            xt = xt_ns,
            ct = ct_ns)

def nonlinear_gau_cins(Nclass=20):
    """
    Crucial difference is latents are conditionally independent
    """
    lags = 2
    Nlayer = 3
    length = 4
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 7500
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "nonlinear_gau_cins_%d"%Nclass)
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
    yt = []; xt = []; ct = []
    yt_ns = []; xt_ns = []; ct_ns = []
    modMat = np.random.uniform(0, 1, (latent_size, Nclass))
    # Mixing function
    for j in range(Nclass):
        ct.append(j * np.ones(batch_size))
        y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
        y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
        
        # Initialize the dataset
        for i in range(lags):
            yt.append(y_l[:,i,:])
        mixedDat = np.copy(y_l)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_l = np.copy(mixedDat)
        for i in range(lags):
            xt.append(x_l[:,i,:])
        # Generate time series dataset
        for i in range(length):
            # Transition function
            y_t = np.random.normal(0, 0.1, (batch_size, latent_size))
            # y_t = np.random.laplace(0, 0.1, (batch_size, latent_size))
            y_t = np.multiply(y_t, modMat[:, j])

            for l in range(lags):
                # y_t += np.tanh(np.dot(y_l[:,l,:], transitions[l]))
                y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
            y_t = leaky_ReLU(y_t, negSlope)
            yt.append(y_t)

            # Mixing function
            mixedDat = np.copy(y_t)
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingList[l])
            x_t = np.copy(mixedDat)
            xt.append(x_t)

            y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]
        
        yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); ct = np.array(ct).transpose(1,0)
        yt_ns.append(yt); xt_ns.append(xt); ct_ns.append(ct)
        yt = []; xt = []; ct = []

    yt_ns = np.vstack(yt_ns)
    xt_ns = np.vstack(xt_ns)
    ct_ns = np.vstack(ct_ns)

    np.savez(os.path.join(path, "data"), 
            yt = yt_ns, 
            xt = xt_ns,
            ct = ct_ns)

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)

def nonlinear_gau_cins_sparse():
    """
    Crucial difference is latents are conditionally independent
    """
    lags = 2
    Nlayer = 3
    length = 4
    Nclass = 20
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 7500
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "nonlinear_gau_cins_sparse")
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

    mask = controlable_sparse_transitions(latent_size, lags, sparsity=0.3)
    for l in range(lags):
        transitions[l] = transitions[l] * mask
        
    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)

    yt = []; xt = []; ct = []
    yt_ns = []; xt_ns = []; ct_ns = []
    modMat = np.random.uniform(0, 1, (latent_size, Nclass))
    # Mixing function
    for j in range(Nclass):
        ct.append(j * np.ones(batch_size))
        y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
        y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
        
        # Initialize the dataset
        for i in range(lags):
            yt.append(y_l[:,i,:])
        mixedDat = np.copy(y_l)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_l = np.copy(mixedDat)
        for i in range(lags):
            xt.append(x_l[:,i,:])
        # Generate time series dataset
        for i in range(length):
            # Transition function
            y_t = np.random.normal(0, 0.1, (batch_size, latent_size))
            # y_t = np.random.laplace(0, 0.1, (batch_size, latent_size))
            y_t = np.multiply(y_t, modMat[:, j])

            for l in range(lags):
                # y_t += np.tanh(np.dot(y_l[:,l,:], transitions[l]))
                y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
            y_t = leaky_ReLU(y_t, negSlope)
            yt.append(y_t)

            # Mixing function
            mixedDat = np.copy(y_t)
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingList[l])
            x_t = np.copy(mixedDat)
            xt.append(x_t)

            y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]
        
        yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); ct = np.array(ct).transpose(1,0)
        yt_ns.append(yt); xt_ns.append(xt); ct_ns.append(ct)
        yt = []; xt = []; ct = []

    yt_ns = np.vstack(yt_ns)
    xt_ns = np.vstack(xt_ns)
    ct_ns = np.vstack(ct_ns)

    np.savez(os.path.join(path, "data"), 
            yt = yt_ns, 
            xt = xt_ns,
            ct = ct_ns)

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)

def instan_temporal():
    lags = 1
    Nlayer = 3
    length = 4
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    noise_scale = 0.1
    batch_size = 50000
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "instan_temporal")
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
    # Zt = f(Zt-1, et) + AZt
    for i in range(length):
        # Transition function
        y_t = torch.distributions.laplace.Laplace(0,noise_scale).rsample((batch_size, latent_size)).numpy()
        # y_t = (y_t - np.mean(y_t, axis=0 ,keepdims=True)) / np.std(y_t, axis=0 ,keepdims=True)
        for l in range(lags):
            y_t += np.dot(y_l[:,l,:], transitions[l])
            y_t = leaky_ReLU(y_t, negSlope) # f(Zt-1, et) with LeakyRelu as AVF
            y_t += np.dot(y_t, transitions[l])
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


def case1_dependency():
    lags = 2
    Nlayer = 3
    length = 4
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 7500
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "case1_dependency")
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

    # create DAG randomly
    import networkx as nx
    from random import randint, random
    def random_dag(nodes: int, edges: int):
        """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
        G = nx.DiGraph()
        for i in range(nodes):
            G.add_node(i)
        while edges > 0:
            a = randint(0, nodes-1)
            b = a
            while b == a:
                b = randint(0, nodes-1)
            G.add_edge(a, b)
            if nx.is_directed_acyclic_graph(G):
                edges -= 1
            else:
                # we closed a loop!
                G.remove_edge(a, b)
        return G
    DAG = random_dag(latent_size, 40)
    dag = nx.to_numpy_array(DAG)

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)

    yt = []; xt = []
    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
    
    # Initialize the dataset
    for i in range(lags):
        yt.append(y_l[:,i,:])
    mixedDat = np.copy(y_l)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])
    # Generate time series dataset
    for i in range(length):
        # Transition function
        # y_t = np.random.normal(0, 0.1, (batch_size, latent_size))
        y_t = np.random.laplace(0, 0.1, (batch_size, latent_size))

        for l in range(lags):
            # y_t += np.tanh(np.dot(y_l[:,l,:], transitions[l]))
            y_t += np.dot(y_l[:,l,:], transitions[l])
        y_t = np.dot(y_t, np.ones((latent_size,latent_size))-dag)
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


def case2_nonstationary_causal():
    lags = 2
    Nlayer = 3
    length = 4
    Nclass = 20
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 7500
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "case2_nonstationary_causal")
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

    yt = []; xt = []; ct = []
    yt_ns = []; xt_ns = []; ct_ns = []
    # Mixing function
    for j in range(Nclass):
        ct.append(j * np.ones(batch_size))

        masks = create_sparse_transitions(latent_size, lags, j)
        for l in range(lags):
            transitions[l] = transitions[l] * masks[l]

        y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
        y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
        
        # Initialize the dataset
        for i in range(lags):
            yt.append(y_l[:,i,:])
        mixedDat = np.copy(y_l)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_l = np.copy(mixedDat)
        for i in range(lags):
            xt.append(x_l[:,i,:])
        # Generate time series dataset
        for i in range(length):
            # Transition function
            y_t = np.random.normal(0, 0.1, (batch_size, latent_size))

            for l in range(lags):
                # y_t += np.tanh(np.dot(y_l[:,l,:], transitions[l]))
                y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
            y_t = leaky_ReLU(y_t, negSlope)
            yt.append(y_t)

            # Mixing function
            mixedDat = np.copy(y_t)
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingList[l])
            x_t = np.copy(mixedDat)
            xt.append(x_t)

            y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]
        
        yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); ct = np.array(ct).transpose(1,0)
        yt_ns.append(yt); xt_ns.append(xt); ct_ns.append(ct)
        yt = []; xt = []; ct = []
        
        for l in range(lags):
            B = transitions[l]
            np.save(os.path.join(path, "W%d%d"%(j, lags-l)), B)

    yt_ns = np.vstack(yt_ns)
    xt_ns = np.vstack(xt_ns)
    ct_ns = np.vstack(ct_ns)

    np.savez(os.path.join(path, "data"), 
            yt = yt_ns, 
            xt = xt_ns,
            ct = ct_ns)


def gen_da_data_ortho(Nsegment, varyMean=False, seed=1):
    """
    generate multivariate data based on the non-stationary non-linear ICA model of Hyvarinen & Morioka (2016)
    we generate mixing matrices using random orthonormal matrices
    INPUT
        - Ncomp: number of components (i.e., dimensionality of the data)
        - Nlayer: number of non-linear layers!
        - Nsegment: number of data segments to generate
        - NsegmentObs: number of observations per segment
        - source: either Laplace or Gaussian, denoting distribution for latent sources
        - NonLin: linearity employed in non-linear mixing. Can be one of "leaky" = leakyReLU or "sigmoid"=sigmoid
          Specifically for leaky activation we also have:
            - negSlope: slope for x < 0 in leaky ReLU
            - Niter4condThresh: number of random matricies to generate to ensure well conditioned
    OUTPUT:
      - output is a dictionary with the following values:
        - sources: original non-stationary source
        - obs: mixed sources
        - labels: segment labels (indicating the non stationarity in the data)
    """
    path = os.path.join(root_dir, "da_gau_%d"%Nsegment)
    os.makedirs(path, exist_ok=True)
    Ncomp = 4
    Ncomp_s = 1
    Nlayer = 3
    NsegmentObs = 7500
    negSlope = 0.2
    NonLin = 'leaky'
    source = 'Gaussian'
    np.random.seed(seed)
    # generate non-stationary data:
    Nobs = NsegmentObs * Nsegment  # total number of observations
    labels = np.array([0] * Nobs)  # labels for each observation (populate below)

    # generate data, which we will then modulate in a non-stationary manner:
    if source == 'Laplace':
        dat = np.random.laplace(0, 1, (Nobs, Ncomp))
        dat = scale(dat)  # set to zero mean and unit variance
    elif source == 'Gaussian':
        dat = np.random.normal(0, 1, (Nobs, Ncomp))
        dat = scale(dat)
    else:
        raise Exception("wrong source distribution")

    # get modulation parameters
    modMat = np.random.uniform(0.01, 3, (Ncomp_s, Nsegment))

    if varyMean:
        meanMat = np.random.uniform(-3, 3, (Ncomp_s, Nsegment))
    else:
        meanMat = np.zeros((Ncomp_s, Nsegment))
    # now we adjust the variance within each segment in a non-stationary manner
    for seg in range(Nsegment):
        segID = range(NsegmentObs * seg, NsegmentObs * (seg + 1))
        dat[segID, -Ncomp_s:] = np.multiply(dat[segID, -Ncomp_s:], modMat[:, seg])
        dat[segID, -Ncomp_s:] = np.add(dat[segID, -Ncomp_s:], meanMat[:, seg])
        labels[segID] = seg

    # now we are ready to apply the non-linear mixtures:
    mixedDat = np.copy(dat)

    # generate mixing matrices:
    # now we apply layers of non-linearity (just one for now!). Note the order will depend on natural of nonlinearity!
    # (either additive or more general!)
    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(Ncomp)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)

        # we first apply non-linear function, then causal matrix!
        if NonLin == 'leaky':
            mixedDat = leaky_ReLU(mixedDat, negSlope)
        elif NonLin == 'sigmoid':
            mixedDat = sigmoidAct(mixedDat)
        # apply mixing:
        mixedDat = np.dot(mixedDat, A)

    np.savez(os.path.join(path, "data"), 
             y = dat, 
             x = mixedDat,
             c = labels)

if __name__ == "__main__":
    # linear_nonGaussian()
    # linear_nonGaussian_ts()
    # nonlinear_Gaussian_ts()
    # nonlinear_nonGaussian_ts()
    # nonlinear_ns()
    # nonlinear_gau_ns()
    # case1_dependency()
    # case2_nonstationary_causal()
    # nonlinear_gau_cins_sparse()
    # instan_temporal()
    # for Nclass in  [1, 5, 10, 15, 20]:
    #     nonlinear_gau_cins(Nclass)
    gen_da_data_ortho(Nsegment=5, varyMean=True)
