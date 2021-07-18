# Input: `{f_i}_i=1^T`: [BS, len = T, dim = 8]
# Output: `{z_i}_i=1^T`: [BS, len = T, dim = 8]
# Bidirectional GRU/LSTM (1 layer)
# Sequential sampling & reparametrize

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

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

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

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class Inference_Net(nn.Module):
    def __init__(self, input_dim=8, z_dim=8, hidden_dim=128):
        super(Inference_Net, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        '''
        # 1. encoder & decoder parts
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
        self.encode_rnn = nn.LSTM(input_dim + z_dim, z_dim, num_layers=1, batch_first=True)
        self.predict_rnn = nn.LSTM(z_dim*2, z_dim, num_layers=1, batch_first=True)
        self.decode_rnn = nn.LSTM(z_dim*2, z_dim, num_layers=1, batch_first=True)

        # Beta
        self.beta_mu_layer = nn.Linear(z_dim, z_dim)
        self.beta_sigma_layer = nn.Linear(z_dim, z_dim)

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
        latent = []
        for i in range(mu.shape[1]):
            input_beta = self.pyro_sample('input_beta', dist.Normal, mu[:,i,:], sigma[:,i,:], sample)
            beta = input_beta.view(-1, 1, self.z_dim)
            latent.append(beta)
        
        latent = torch.stack(latent, dim=1).view(mu.shape[0], -1, self.z_dim)
        return latent

    def forward(self, xt, lag, sample=True):
        # encoder
        _, length, latent_dim  = xt.shape
        xt = self.encoder(xt.view(-1, latent_dim))
        xt = xt.view(-1, length, latent_dim)
        ## lstm encoder
        encoder_outputs, hidden_states = self._encode(xt, lag)
        ## lstm decoder
        pred_outputs, pred_beta_mu, pred_beta_sigma = self._decode(encoder_outputs, hidden_states)
        ## sample latent
        mu = pred_beta_mu
        sigma = pred_beta_sigma
        latent = self.sample_latent(mu, sigma, sample)
        # decoder
        decoded_output = self.decoder(latent)
        return decoded_output, mu, sigma, latent
        
    def _encode(self, xt, lag):
        hidden = None
        frame_outputs = []
        hidden_states = []
        encoder_outputs = [] 
        batch_size, length, latent_dim  = xt.shape
        prev_hidden = Variable(torch.zeros(batch_size, 1, latent_dim).cuda())

        for i in range(length):
            rnn_input = torch.cat([xt[:, i:i+1, :], prev_hidden], dim=2)
            output, hidden = self.encode_rnn(rnn_input, hidden)
            h = hidden[0]; c= hidden[1]
            prev_hidden= h.view(batch_size, 1, -1)
            frame_outputs.append(output)
            hidden_states.append((h, c))
          
        encoder_outputs = torch.squeeze(torch.stack(frame_outputs, dim=1))

        return encoder_outputs, hidden_states

    def _decode(self, encoder_outputs, hidden_states):
        pred_outputs = []
        frame_outputs = []
        hidden_states.reverse()
        pred_beta_mu, pred_beta_sigma = None, None
        batch_size, len, latent_dim  = encoder_outputs.shape
        encoder_outputs = torch.flip(encoder_outputs, dims=[1])
        prev_hidden = Variable(torch.zeros(batch_size, 1, latent_dim).cuda())

        for i in range(len):
            hidden = hidden_states[i]
            prev_outputs = encoder_outputs[:, i:i+1, :]
            rnn_input = torch.cat([prev_outputs, prev_hidden], dim=2)
            output, hidden = self.predict_rnn(rnn_input, hidden)
            prev_hidden= hidden[0].view(batch_size, 1, -1)
            frame_outputs.append(output)
        # pdb.set_trace()
        pred_outputs = torch.flip(torch.squeeze(torch.stack(frame_outputs, dim=1)), dims=[1])

        pred_beta_mu = self.beta_mu_layer(pred_outputs)
        pred_beta_sigma = self.beta_sigma_layer(pred_outputs)
        pred_beta_sigma = F.softplus(pred_beta_sigma)

        return pred_outputs, pred_beta_mu, pred_beta_sigma
