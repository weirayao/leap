import torch
import torch.nn as nn
import copy

from typing import Tuple

class GroupLinearLayer(nn.Module):
    """GroupLinearLayer computes N dinstinct linear transformations at once"""
    def __init__(
        self, 
        din: int, 
        dout: int, 
        num_blocks: int) -> None:
        """Group Linear Layer module

        Args:
            din: The feature dimension of input data.
            dout: The projected dimensions of data.
            num_blocks: The number of linear transformation to compute at once.
        """
        super(GroupLinearLayer, self).__init__()
        self.w = nn.Parameter(0.01 * torch.randn(num_blocks, din, dout))

    def forward(
        self,
        x: torch.Tensor) -> torch.Tensor:
        # x: [BS,num_blocks,din]->[num_blocks,BS,din]
        x = x.permute(1,0,2)
        x = torch.bmm(x, self.w)
        # x: [BS,num_blocks,dout]
        x = x.permute(1,0,2)
        return x

class LinearSSMCell(nn.Module):
    """LinearSSMCell performs linear state space model operation for one time step
       
       y_t = \sum_{p=1}^L B_p @ y_{t-p} + \epsilon_t
       x_t = Ay_t

    Assume x_t and y_t are of the same dimesion.

    Args:
        hidden_size: The number of latent causal factors.
        time_lags: Past time lags to consider for VAR.
        output_size: The size of observation x_t.
    """
    def __init__(
        self, 
        hidden_size: int, 
        time_lags: int, 
        output_size: int) -> None:
        """Constructs LinearSSMCell"""
        super(LinearSSMCell, self).__init__()
        self.hidden_size = hidden_size
        self.time_lags = time_lags
        self.mu = GroupLinearLayer(din = hidden_size, 
                                   dout = hidden_size, 
                                   num_blocks = time_lags)
        self.mix = nn.Linear(hidden_size, output_size)

    def forward(
        self, 
        hiddens: torch.Tensor, 
        noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hiddens: Past latent causal factors [BS,time_lags,hidden_size]. 
        mu = self.mu(hiddens)
        mu = torch.sum(mu, dim=1)
        y = mu + noise
        x = self.mix(y)
        # Update hiddens to the most recent ones
        hiddens = torch.cat((y, hiddens[:,:-1,:]), dim=1)
        return x, hiddens

    def init_hidden(self, bsz: int = 32):
        return nn.init.kaiming_uniform_(torch.empty(bsz, self.time_lags, self.hidden_size))


class MBDCell(nn.Module):
    """Multichannel Blind Deconvolution (MBD) to recover noise.
    
                \epsilon_t = \sum_{p=0}^L W_l @ x_{t-k}

    Args:
        output_size: The size of observation x_t.
        hidden_size: The number of latent causal factors.
        time_lags: Past time lags to consider for MBD.
    """
    def __init__(
        self,
        output_size: int,
        hidden_size: int, 
        time_lags: int) -> None:
        """"Constructs MBDCell"""
        super(MBDCell, self).__init__()
        num_blocks = time_lags + 1
        self.dconv = GroupLinearLayer(din = output_size, 
                                      dout = hidden_size, 
                                      num_blocks = num_blocks)
    
    def forward(
        self, 
        x_t: torch.Tensor,
        x_p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decovolution function
        
        Args:
            x_t: The observation at current time step.
            x_p: The observations for past time steps.

        Returns:
            e: The noise terms at current time step.
            x_p: The updated past observations.
        """
        # x: [BS, time_lags+1, output_size]
        x = torch.cat((x_t, x_p), dim=1)
        e = self.dconv(x)
        e = torch.sum(e, dim=1)
        # Update x_p to most recent ones
        x_p = x[:,:-1,:]
        return e, x_p


