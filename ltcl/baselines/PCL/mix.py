import torch
import torch.nn as nn


class MixingMLP(nn.Module):
    """The invertible mixing function with multilayer perceptron"""
    def __init__(
        self, 
        input_dims: int, 
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
        
        self.layers = nn.Sequential(*self.layers)
    
    def forward(
        self, 
        x: torch.Tensor) -> torch.Tensor:
        """Returns mixed observations from sources"""
        return self.layers(x)


class MixingGNN(nn.Module):
    pass
    
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