# Require: input_dim, z_dim, hidden_dim
# Input: {f_i}_i=1^T: [BS, len=T, dim=8]
# Output: {z_i}_i=1^T: [BS, len=T, dim=8]
# Bidirectional GRU/LSTM (1 layer)
# Sequential sampling & reparameterization

import pyro
import torch
import ipdb as pdb
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import pyro.distributions as dist
from collections import defaultdict
from torch.autograd import Variable


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


class Inference_Net(nn.Module):
    def __init__(self, input_dim=8, z_dim=8, hidden_dim=128):
        super(Inference_Net, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(z_dim, z_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(z_dim, z_dim, num_layers=1, batch_first=True, bidirectional=True)
        '''
        # 1. encoder & decoder (weiran parts)
        # input: {xi}_{i=1}^T; output: {fi}_{i=1}^T
        # input: {zi}_{i=1}^T; output: {recon_xi}_{i=1}^T

        self.encoder = nn.Sequential(
                                       nn.Linear(input_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, z_dim),
                                       nn.LeakyReLU(0.2)
                                    )
        self.decoder = nn.Sequential(
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(z_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, input_dim)
                                    )
        '''
        self.mu_sample = nn.Sequential(
                                       nn.Linear(3*z_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, z_dim),
                                       nn.LeakyReLU(0.2)
                                    )
        self.var_sample = nn.Sequential(
                                       nn.Linear(3*z_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, z_dim),
                                       nn.Softmax(0.2),
                                    )

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def pyro_sample(self, name, fn, mu, sigma, sample=True):
        '''
        Sample with pyro.sample. fn should be dist.Normal.
        If sample is False, then return mean.
        '''
        if sample:
            return pyro.sample(name, fn(mu, sigma))
        else:
            return mu.contiguous()

    def sample_latent(self, mu, sigma, sample):
        latent = self.pyro_sample('input_beta', dist.Normal, mu, sigma, sample).view(-1, self.z_dim)
        return latent

    def forward(self, ft, sample=True):
        ''' 
        ## encoder (weiran part)
        # input: xt(batch, seq_len, z_dim)
        # output: ft(seq_len, batch, z_dim)
        _, length, _  = xt.shape
        ft = self.encoder(xt.view(-1, self.z_dim))
        ft = ft.view(-1, length, self.z_dim)
        '''

        ## bidirectional lstm/gru 
        # input: ft(seq_len, batch, z_dim)
        # output: beta(batch, seq_len, z_dim)
        hidden = None
        beta, hidden = self.lstm(ft, hidden)
        # beta, hidden = self.gru(ft, hidden)
        
        ## sequential sampling & reparametrization
        ## transition: p(zt|z_tau)
        latent = []; mu = []; sigma = []
        init = torch.zeros(beta.shape)
        for i in range(beta.shape[1]):
            if i == 0:
                input = torch.cat([init[:,i,:], init[:,i,:], beta[:,i,:]], dim=1)
            elif i == 1:
                input = torch.cat([init[:,i,:], latent[-1], beta[:,i,:]], dim=1)
            else:
                input = torch.cat([latent[-2], latent[-1], beta[:,i,:]], dim=1)
            mut = self.mu_sample(input)
            sigmat = self.var_sample(input)
            latentt = self.sample_latent(mut, sigmat, sample)
            latent.append(latentt)
            mu.append(mut); sigma.append(sigmat)
        
        latent = torch.squeeze(torch.stack(latent, dim=1))
        mu = torch.squeeze(torch.stack(mu, dim=1))
        sigma = torch.squeeze(torch.stack(sigma, dim=1))

        '''
        ## decoder (weiran part)
        # input: latent(batch, seq_len, z_dim)
        # output: recon_xt(batch, seq_len, z_dim)
        recon_xt = self.decoder(latent.view(-1, self.z_dim))
        recon_xt = recon_xt.view(-1, length, self.z_dim)
        '''

        return latent, mu, sigma
