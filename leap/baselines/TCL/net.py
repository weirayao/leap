"""net.py"""

import torch
import numpy as np
from torch import nn
import torch.nn.init as init
from torch.nn import functional as F
from torch import distributions as dist


class TCLMLP(nn.Module):
    def __init__(self, 
                 input_dim=8, z_dim=8, 
                 hidden_dim=128, nclass=20):
        super(TCLMLP, self).__init__()
        self.z_dim = z_dim
        self.nclass = nclass
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.net_feats = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, z_dim),
            nn.LeakyReLU(0.2, True)
        )

        self.net_logits = nn.Sequential(
            nn.Linear(z_dim, self.nclass),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, x):
        feats = self.net_feats(x)
        feats = torch.abs(feats)
        logits = self.net_logits(feats)
        
        return logits, feats

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)