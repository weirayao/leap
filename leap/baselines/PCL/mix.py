import torch
import torch.nn as nn
import numpy as np
from leap.modules.components.keypoint import SpatialSoftmax

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
        
class MixingMLP(nn.Module):
    """The invertible mixing function with multilayer perceptron"""
    def __init__(
        self, 
        input_dims: int, 
        z_dim: int, 
        num_layers: int = 3, 
        negative_slope: float = 0.01
    ) -> None:
        """Construct a mixing function
        
        Args:
            input_dims: The feature dimension of input data.
            num_layers: The numberof layers in MLP.
            negative_slope: The slope of negative region in LeakyReLU.
        """
        super(MixingMLP, self).__init__()
        self.layers = [ ]
        for _ in range(num_layers):
            self.layers.append(nn.Linear(input_dims, input_dims))
            self.layers.append(nn.LeakyReLU(negative_slope))
        self.layers.append(nn.Linear(input_dims, z_dim))
        self.layers = nn.Sequential(*self.layers)
    
    def forward(
        self, 
        x: torch.Tensor) -> torch.Tensor:
        """Returns mixed observations from sources"""
        return self.layers(x)


class MixingCNN(nn.Module):
    """"""
    def __init__(self, z_dim=10, nc=3, hidden_dim=256):
        super().__init__()
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
            nn.Linear(hidden_dim, z_dim),             # B, z_dim*2
        )

    def forward(self, x):
        return self.encoder(x)

class MixingKP(nn.Module):
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

    def forward(self, x):
        heatmap = self.encoder(x)
        batch_size = heatmap.shape[0]
        mu = self.integrater(heatmap)
        mu = mu.view(batch_size, -1)
        return mu

class ScoringFunc(nn.Module):

    def __init__(self, input_dims=2, hidden_dims=128, num_layers=3):
        super(ScoringFunc, self).__init__()
        self.layers = [ ]
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dims, hidden_dims))
                self.layers.append(nn.ReLU())
            elif i == num_layers-1:
                self.layers.append(nn.Linear(hidden_dims, 1))
            else:
                self.layers.append(nn.Linear(hidden_dims, hidden_dims))
                self.layers.append(nn.ReLU()) 

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)