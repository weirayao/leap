"""model.py"""

import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class SlowVAEMLP(nn.Module):
    """Encoder and Decoder architecture for 2D Shapes data."""
    def __init__(self, input_dim=8, z_dim=8, hidden_dim=128):
        super(SlowVAEMLP, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, 2*z_dim),
            nn.LeakyReLU(0.2, True)
        )
        self.decoder = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, input_dim)
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=False):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)
    
    def _decode(self, z):
        return self.decoder(z)


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