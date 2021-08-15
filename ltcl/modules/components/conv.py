import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class ConvDecoder(nn.Module):
    """Convolutional decoder for beta-VAE"""
    def __init__(self, z_dim=10, nc=3, hidden_dim=256):
        super().__init__()
        self.upsample = nn.Sequential(
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
                                        nn.ConvTranspose2d(32, nc, 4, 2, 1)  # B, nc, 64, 64
                                     ) 
    def forward(self, x):
        return self.upsample(x)

class ConvEncoder(nn.Module):

    def __init__(self, z_dim=10, nc=3, hidden_dim=256):
        super().__init__()
        self.downsample = nn.Sequential(
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
                                        nn.Conv2d(64, hidden_dim, 4, 1),     # B, hidden_dim,  1,  1
                                        nn.BatchNorm2d(hidden_dim),
                                        nn.ReLU(True),
                                        View((-1, hidden_dim*1*1)),       # B, hidden_dim
                                        nn.Linear(hidden_dim, z_dim)             # B, z_dim
                                        )
    def forward(self, x):
        return self.downsample(x)