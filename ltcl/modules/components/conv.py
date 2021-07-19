import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class ConvDecoder(nn.Module):
    """Convolutional decoder for beta-VAE"""
    def __init__(self, latent_dims=128):
        super(ConvDecoder, self).__init__()
        self.fc = nn.Linear(latent_dims, 2048)
        self.upsample = nn.Sequential(
                        nn.ConvTranspose2d(512,256,4,2),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.ConvTranspose2d(256,128,4,2),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.ConvTranspose2d(128,64,4,2),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.ConvTranspose2d(64,32,4,2),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.ConvTranspose2d(32,16,4,2),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),
                        nn.ConvTranspose2d(16,3,4,2)
                        )
                        
        self.resize = transforms.Resize((64,64))
        
    def forward(self, x):
        x = F.relu(self.fc(x))
        x = torch.reshape(x, (x.shape[0], 512, 2, 2))
        x = self.upsample(x)
        x = self.resize(x)
        return x