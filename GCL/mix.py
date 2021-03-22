import torch
import torch.nn as nn


class MixingMLP(nn.Module):

    def __init__(self, input_dims: int, num_layers=3, negative_slope=0.01):
        super(MixingMLP, self).__init__()
        self.layers = [ ]
        for _ in range(num_layers):
            self.layers.append(nn.Linear(input_dims, input_dims))
            self.layers.append(nn.LeakyReLU(negative_slope))
        
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.layers(x)


class MixingGNN(nn.Module):
    



class FeatureMLP(nn.Module):

    def __init__(self, input_dims: int, num_layers=3, negative_slope=0.01):
        super(FeatureMLP, self).__init__()
        self.layers = [ ]
        for _ in range(num_layers):
            self.layers.append(nn.Linear(input_dims, input_dims))
            self.layers.append(nn.LeakyReLU(negative_slope))
        
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.layers(x)

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