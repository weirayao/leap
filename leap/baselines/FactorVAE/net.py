"""model.py"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from leap.modules.components.keypoint import SpatialSoftmax

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
        
class Discriminator(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, 2),
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

    def forward(self, z):
        return self.net(z).squeeze()


class FactorVAEMLP(nn.Module):
    """Encoder and Decoder architecture for 2D Shapes data."""
    def __init__(self, input_dim=8, z_dim=8, hidden_dim=128):
        super(FactorVAEMLP, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encode = nn.Sequential(
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
        self.decode = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, input_dim)
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

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode(z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()

class FactorVAECNN(nn.Module):
    """Encoder and Decoder architecture for KittMask data."""
    def __init__(self, z_dim=10, nc=3, hidden_dim=256):
        super(FactorVAECNN, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.nc = nc
        self.encode = nn.Sequential(
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

        self.decode = nn.Sequential(
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

    def weight_init(self, mode='kaiming'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode(z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()

class FactorVAEKP(nn.Module):
    """Visual Encoder/Decoder for Ball dataset."""
    def __init__(self, k=5, nc=3, nf=16, norm_layer='Batch'):
        super().__init__()
        self.nc = nc
        self.k = k
        self.z_dim = self.k * 2
        height = 64
        width = 64
        lim=[-1., 1., -1., 1.]
        self.height = height
        self.width = width
        self.lim = lim
        x = np.linspace(lim[0], lim[1], width // 4)
        y = np.linspace(lim[2], lim[3], height // 4)
        z = np.linspace(-1., 1., k)
        self.register_buffer('x', torch.FloatTensor(x))
        self.register_buffer('y', torch.FloatTensor(y))
        self.register_buffer('z', torch.FloatTensor(z))

        self.integrater = SpatialSoftmax(height=height//4, width=width//4, channel=k, lim=lim)
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nf, 7, 1, 3),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf, 5, 1, 2),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 4) x 16 x 16
            nn.Conv2d(nf * 4, k, 1, 1)
        )

        self.fc = nn.Sequential(View((-1, k * 16 * 16)),
                                nn.Linear(k * 16 * 16, self.z_dim)
                                )

        self.decoder = nn.Sequential(            
            nn.ConvTranspose2d(self.k, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 4) x 32 x 32
            nn.Conv2d(nf * 4, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 4) x 32 x 32
            nn.ConvTranspose2d(nf * 2, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 2) x 64 x 64
            nn.Conv2d(nf * 2, nf, 5, 1, 2),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 2) x 64 x 64
            nn.Conv2d(nf, 3, 7, 1, 3))

    def forward(self, x, no_dec=False):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)
        x_recon = self._decode(z)

        if no_dec:
            return z.squeeze()
        else:
            return x_recon, mu, logvar, z.squeeze()

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def keypoint_to_heatmap(self, keypoint, inv_std=10.):
        # keypoint: B x n_kp x 2
        # heatpmap: B x n_kp x (H / 4) x (W / 4)
        # ret: B x n_kp x (H / 4) x (W / 4)
        height = self.height // 4
        width = self.width // 4

        mu_x, mu_y = keypoint[:, :, :1].unsqueeze(-1), keypoint[:, :, 1:].unsqueeze(-1)
        y = self.y.view(1, 1, height, 1)
        x = self.x.view(1, 1, 1, width)

        g_y = (y - mu_y)**2
        g_x = (x - mu_x)**2
        dist = (g_y + g_x) * inv_std**2

        hmap = torch.exp(-dist)

        return hmap

    def _encode(self, x):
        heatmap = self.encoder(x)
        batch_size = heatmap.shape[0]
        mu = self.integrater(heatmap)
        mu = mu.view(batch_size, -1)
        logvar = self.fc(heatmap)
        return torch.cat((mu,logvar), dim=-1)

    def _decode(self, z):
        kpts = z.view(-1, self.k, 2)
        hmap = self.keypoint_to_heatmap(kpts)
        return self.decoder(hmap)

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
