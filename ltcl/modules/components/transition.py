"""Prior Network"""
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
from ltcl.modules.components.mlp import NLayerLeakyMLP
from ltcl.modules.components.graph import PropNet
from ltcl.modules.components.transforms import AfflineCoupling
from ltcl.modules.components.base import GroupLinearLayer
import ipdb as pdb

class LinearTransitionPrior(nn.Module):
    # Deprecated
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
    
class MBDTransitionPrior(nn.Module):

    def __init__(self, lags, latent_size, bias=False):
        super().__init__()
        # self.init_hiddens = nn.Parameter(0.001 * torch.randn(lags, latent_size))    
        # out[:,:,0] = (x[:,:,0]@conv.weight[:,:,0].T)+(x[:,:,1]@conv.weight[:,:,1].T) 
        # out[:,:,1] = (x[:,:,1]@conv.weight[:,:,0].T)+(x[:,:,2]@conv.weight[:,:,1].T)
        self.L = lags      
        self.transition = GroupLinearLayer(din = latent_size, 
                                           dout = latent_size, 
                                           num_blocks = lags,
                                           diagonal = False)
    
    def forward(self, x, mask=None):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        batch_size, length, input_dim = x.shape
        # init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
        # x = torch.cat((init_hiddens, x), dim=1)
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape

        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        residuals = torch.sum(self.transition(yy), dim=1) - xx.squeeze()
        residuals = residuals.reshape(batch_size, -1, input_dim)
        # Dummy jacobian matrix (0) to represent identity mapping
        log_abs_det_jacobian = torch.zeros(batch_size, device=x.device)
        return residuals, log_abs_det_jacobian

class PNLTransitionPrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size, 
        num_layers=3, 
        hidden_dim=64):
        super().__init__()
        self.L = lags
        # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))       
        self.f1 = NLayerLeakyMLP(in_features=lags*latent_size, 
                                 out_features=latent_size, 
                                 num_layers=num_layers, 
                                 hidden_dim=hidden_dim)
        
        # Approximate the inverse of mild invertible function
        self.f2 = AfflineCoupling(n_blocks = num_layers, 
                                  input_size = latent_size, 
                                  hidden_size = hidden_dim, 
                                  n_hidden = 1, 
                                  batch_norm = False)
    
    def forward(self, x, mask=None):
        # x: [BS, T, D]
        batch_size, length, input_dim = x.shape
        # Pad learnable [BS, lags, latent_size] at the front
        x_inv, log_abs_det_jacobian = self.f2(x[:,self.L:,:].reshape(-1, input_dim))
        x_inv = x_inv.reshape(batch_size, length-self.L, input_dim)

        residuals = [ ]
        for t in range(length-self.L):
            x_in = x[:,t:t+self.L,:].view(batch_size, -1)
            res = self.f1(x_in) - x_inv[:,t]
            residuals.append(res)
        residuals = torch.stack(residuals, dim=1)
        # log_abs_det_jacobian: [BS, ]
        log_abs_det_jacobian = torch.sum(log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, log_abs_det_jacobian

#TODO: Markovian Transition Prior (Graph Interaction Network)
class INTransitionPrior(nn.Module):

    def __init__(self):
        raise NotImplementedError


class NPTransitionPrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size, 
        num_layers=3, 
        hidden_dim=64):
        super().__init__()
        self.L = lags
        # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))       
        gs = [NLayerLeakyMLP(in_features=lags*latent_size+1, 
                             out_features=1, 
                             num_layers=num_layers, 
                             hidden_dim=hidden_dim) for i in range(latent_size)]
        
        self.gs = nn.ModuleList(gs)
    
    def forward(self, x):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        batch_size, length, input_dim = x.shape
        # init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
        # x = torch.cat((init_hiddens, x), dim=1)
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        yy = yy.reshape(-1, self.L*input_dim)
        residuals = [ ]
        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            inputs = torch.cat((yy, xx[:,:,i]),dim=-1)
            residual = self.gs[i](inputs)
            pdd = jacobian(self.gs[i], inputs, create_graph=True)
            # Determinant of low-triangular mat is product of diagonal entries
            logabsdet = torch.log(torch.abs(torch.diag(pdd[:,0,:,-1])))
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)
        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        return residuals, sum_log_abs_det_jacobian
