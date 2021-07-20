import torch
import torch.nn as nn

class NLayerLeakyMLP(nn.Module):

    def __init__(self, in_features, out_features, num_layers, hidden_dim=64, bias=True):
        super().__init__()
        layers = [ ]
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(in_features, hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden_dim, out_features))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class MLPEncoder(nn.Module):

    def __init__(self, latent_size, num_layers=4, hidden_dim=64):
        super().__init__()
        self.net = NLayerLeakyMLP(in_features=latent_size, 
                                  out_features=latent_size, 
                                  num_layers=num_layers, 
                                  hidden_dim=hidden_dim)
    
    def forward(self, x):
        return self.net(x)

class MLPDecoder(nn.Module):
    """Ground-truth MLP decoder used for data generation"""
    def __init__(self, latent_size, num_layers=2):
        super().__init__()
        # TODO: Do not use ground-truth decoder architecture 
        layers = [ ]
        for l in range(num_layers):
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Linear(latent_size, latent_size))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.net(z)

if __name__ == '__main__':
    net = NLayerLeakyMLP(3,64,32,128)
    print(net)