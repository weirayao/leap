"""Prior Network"""
import torch
import torch.nn as nn
from ltcl.modules.components.mlp import NLayerLeakyMLP
from ltcl.modules.components.graph import PropNet
from ltcl.modules.components.transforms import AfflineCoupling

class LinearTransitionPrior(nn.Module):

    def __init__(self, lags, latent_size, bias=False):
        super().__init__()
        self.init_hiddens = nn.Parameter(0.01 * torch.randn(latent_size, lags))
        # out[:,:,0] = (x[:,:,0]@conv.weight[:,:,0].T)+(x[:,:,1]@conv.weight[:,:,1].T) 
        # out[:,:,1] = (x[:,:,1]@conv.weight[:,:,0].T)+(x[:,:,2]@conv.weight[:,:,1].T)      
        self.transition = nn.Conv1d(in_channels = latent_size, 
                                    out_channels = latent_size, 
                                    kernel_size = lags, 
                                    bias=bias, 
                                    padding_mode='zeros')
    
    def forward(self, x, mask=None):
        # x: [BS, T, D]
        x = x.permute(0,2,1).contiguous()
        batch_size, length, _ = x.shape
        init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
        # Pad learnable [BS, latent_size, lags] at the front
        conv_in = torch.cat((init_hiddens, x), dim=-1)
        conv_out = self.transition(conv_in)
        # Drop the last bc no group-truth label
        residuals = conv_out[...,:-1] - x
        residuals = residuals.permute(0,2,1)
        return residuals
    

class PNLTransitionPrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size, 
        num_layers=3, 
        hidden_dim=64):
        super().__init__()
        self.L = lags
        self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))       
        self.f1 = NLayerLeakyMLP(in_features=lags*latent_size, 
                                 out_features=latent_size, 
                                 num_layers=num_layers, 
                                 hidden_dim=hidden_dim)
        
        # Approximate the inverse of mild invertible function
        # self.f2 = NLayerLeakyMLP(in_features=latent_size, 
        #                          out_features=latent_size, 
        #                          num_layers=1, 
        #                          hidden_dim=hidden_dim)

        self.f2 = AfflineCoupling(n_blocks = 2, 
                                  input_size = latent_size, 
                                  hidden_size = hidden_dim, 
                                  n_hidden = num_layers, 
                                  batch_norm = True)
    
    def forward(self, x, mask=None):
        # x: [BS, T, D]
        batch_size, length, input_dim = x.shape
        init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
        if mask:
            mask = mask.repeat(batch_size, 1)
        # Pad learnable [BS, lags, latent_size] at the front
        x_pad = torch.cat((init_hiddens, x), dim=1)
        x_inv, _ = self.f2(x.view(-1, input_dim))
        x_inv = x_inv.reshape(batch_size, length, input_dim)

        residuals = [ ]
        for t in range(length):
            x_in = x_pad[:,t:t+self.L,:].view(batch_size, -1)
            if mask:
                x_in = x_in * mask
            res = self.f1(x_in) - x_inv[:,t]
            residuals.append(res)
        residuals = torch.stack(residuals, dim=1)
        return residuals

#TODO: Markovian Transition Prior (Graph Interaction Network)
class INTransitionPrior(nn.Module):

    def __init__(self):
        raise NotImplementedError

