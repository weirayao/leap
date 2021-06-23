"""model.py"""

import torch
import ipdb as pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from .conv import ConvDecoder

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

class BetaVAE_CNN(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=3, hidden_dim=256):
        super(BetaVAE_CNN, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, hidden_dim, 4, 1),            # B, hidden_dim,  1,  1
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            View((-1, hidden_dim*1*1)),                 # B, hidden_dim
            nn.Linear(hidden_dim, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),               # B, hidden_dim
            View((-1, hidden_dim, 1, 1)),               # B, hidden_dim,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 64, 4),      # B,  64,  4,  4
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=True):
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

class BetaVAE_Physics(nn.Module):
    """Visual Encoder/Decoder for Ball dataset."""

    def __init__(self, z_dim=32, n_obj=5, nc=3, width=64, height=64, hidden_dim=256):
        super().__init__()
        self.z_dim = z_dim
        self.n_obj = n_obj
        self.nc = nc
        self.width = width
        self.height = height
        # Add coordinate channels
        x_coord, y_coord = self.construct_coord_dims()
        self.register_buffer('x_coord', x_coord)
        self.register_buffer('y_coord', y_coord)

        self.pool = nn.MaxPool2d(2, 2)

        # Visual Encoder Modules
        self.encoder = nn.Sequential(
            nn.Conv2d(nc * 2 + 2, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, hidden_dim, 4, 1),            # B, hidden_dim,  1,  1
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            View((-1, hidden_dim*1*1)),                 # B, hidden_dim
            nn.Linear(hidden_dim, 32),             # B, z_dim*2
        )
        # self.conv1 = nn.Conv2d(nc * 2 + 2, 16, 3, padding=1)
        # self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        # self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        # self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        # self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        # shared linear layer to get pair codes of shape N_obj*cl
        self.fc1 = nn.Linear(32, n_obj * z_dim)
        # shared MLP to encode pairs of pair codes as state codes N_obj*cl
        self.fc2 = nn.Linear(z_dim * 2, z_dim)
        self.fc3 = nn.Linear(z_dim, 2 * z_dim)

        # Verlet integration to unroll the dynamics for 3 frames
        self.dconv = nn.Linear(z_dim, 3 * z_dim)

        # Shared Object Renderer
        # self.renderer = ConvDecoder(z_dim * n_obj)
        self.renderer = nn.Sequential(
            nn.Linear(z_dim * n_obj, hidden_dim),               # B, hidden_dim
            View((-1, hidden_dim, 1, 1)),               # B, hidden_dim,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 64, 4),      # B,  64,  4,  4
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.smoother = nn.Conv2d(3, 3, kernel_size=1, stride=1, bias=True)

    def construct_coord_dims(self):
        """
        Build a meshgrid of x, y coordinates to be used as additional channels
        """
        x = np.linspace(0, 1, self.width)
        y = np.linspace(0, 1, self.height)
        xv, yv = np.meshgrid(x, y)
        xv = np.reshape(xv, [1, 1, self.height, self.width])
        yv = np.reshape(yv, [1, 1, self.height, self.width])
        x_coord = torch.from_numpy(xv.astype('float32'))
        y_coord = torch.from_numpy(yv.astype('float32'))
        return x_coord, y_coord

    def _encode(self, frames):
        """
        Apply visual encoder
        :param frames: Groups of six input frames of shape (BS, T, c, w, h)
        :return: State code distributions of shape (BS, T-2, o, 2*z_dims)
        """
        batch_size, steps, nc, w, h = frames.shape 
        num_obj = self.n_obj
        pairs = []
        for i in range(steps - 1):
            # pair consecutive frames (n, 2c, w, h)
            pair = torch.cat((frames[:, i], frames[:, i+1]), 1)
            pairs.append(pair)

        num_pairs = len(pairs)

        pairs = torch.cat(pairs, 0)
        # add coord channels (n * num_pairs, 2c + 2, w, h)
        x_coord =  self.x_coord.expand(batch_size *  num_pairs, -1,-1,-1)
        y_coord =  self.x_coord.expand(batch_size *  num_pairs, -1,-1,-1)

        pairs = torch.cat([pairs, x_coord, y_coord], dim=1)

        # apply ConvNet to pairs
        # ve_h1 = F.relu(self.conv1(pairs))
        # ve_h1 = self.pool(ve_h1)
        # ve_h2 = F.relu(self.conv2(ve_h1))
        # ve_h2 = self.pool(ve_h2)
        # ve_h3 = F.relu(self.conv3(ve_h2))
        # ve_h3 = self.pool(ve_h3)
        # ve_h4 = F.relu(self.conv4(ve_h3))
        # ve_h4 = self.pool(ve_h4)
        # ve_h5 = F.relu(self.conv5(ve_h4))
        # ve_h5 = self.pool(ve_h5)
        ve_h5 = self.encoder(pairs)
        # pooled to 1x1, 32 channels: (n * num_pairs, 32)
        encoded_pairs = torch.squeeze(ve_h5)
        # final pair encoding (n * num_pairs, o, cl)
        encoded_pairs = self.fc1(encoded_pairs)
        encoded_pairs = encoded_pairs.view(batch_size * num_pairs, num_obj, self.z_dim)
        # chunk pairs encoding, each is (n, o, cl)
        encoded_pairs = torch.chunk(encoded_pairs, num_pairs)

        triples = []
        for i in range(num_pairs - 1):
            # pair consecutive pairs to obtain encodings for triples
            triple = torch.cat([encoded_pairs[i], encoded_pairs[i+1]], 2)
            triples.append(triple)

        # the triples together, i.e. (n, num_pairs - 1, o, 2 * cl)
        triples = torch.stack(triples, 1)
        # apply MLP to triples
        shared_h1 = F.relu(self.fc2(triples))
        distributions = self.fc3(shared_h1)
        return distributions

    def _decode(self, state_codes):
        # state_codes (n, 2, o, z_dims)
        # (n, o, 3*z_dims)
        batch_size, steps, n_objs, z_dims = state_codes.shape
        state_codes = self.dconv(state_codes)
        state_codes = state_codes.view(batch_size, steps, n_objs, 3, z_dims)
        state_codes = state_codes.permute(0,1,3,2,4).contiguous().view(batch_size, steps, 3, n_objs*z_dims)
        state_codes = state_codes.view(-1, n_objs*z_dims)
        frames = self.renderer(state_codes)
        frames = frames.view(batch_size, steps, 3, self.nc, self.height, self.width)

        # frames = frames.view(batch_size, steps, n_objs, 3, self.nc, self.height, self.width)
        # # Sum over object channels (dim=2)
        # frames = torch.sum(frames, dim=2)
        # frames = frames.view(-1, self.nc, self.height, self.width)
        # frames = self.smoother(frames)
        # frames = frames.view(batch_size, steps, 3, self.nc, self.height, self.width)
        return frames

    def forward(self, x, return_z=True):
        distributions = self._encode(x)
        mu = distributions[..., :self.z_dim]
        logvar = distributions[..., self.z_dim:]

        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

class BetaVAE_MLP(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, input_dim=3, z_dim=10, hidden_dim=128):
        super(BetaVAE_MLP, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
                                       nn.Linear(input_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, 2*z_dim)
                                    )
        # Fix the functional form to ground-truth mixing function
        self.decoder = nn.Sequential(
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(z_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, input_dim)
                                    )


        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=True):

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
