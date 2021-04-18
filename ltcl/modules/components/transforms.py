import torch
import torch.nn as nn
from torch.nn import functional as F, init
import numpy as np
from ltcl.modules import components
from .splines import _monotonic_rational_spline
from . import utils
from typing import (Tuple,
                    Optional)

# Invertible Component-wise Spline Transformation #
class ComponentWiseSpline(components.Transform):
    def __init__(self, 
                 input_dim: int, 
                 count_bins: int = 8, 
                 bound: int = 3., 
                 order: str = 'linear'):
        """Component-wise Spline Flow
        Args:
            input_dim: The size of input/latent features.
            count_bins: The number of bins that each can have their own slope.
            bound: Tail bound (outside tail bounds the transformation is identity)
            order: Spline order
        """
        super(ComponentWiseSpline, self).__init__()
        self.input_dim = input_dim
        self.count_bins = count_bins
        self.bound = bound
        self.order = order
        self.unnormalized_widths = nn.Parameter(torch.randn(self.input_dim, self.count_bins))
        self.unnormalized_heights = nn.Parameter(torch.randn(self.input_dim, self.count_bins))
        self.unnormalized_derivatives = nn.Parameter(torch.randn(self.input_dim, self.count_bins - 1))
        # Rational linear splines have additional lambda parameters
        assert self.order in ("linear", "quadratic")
        if self.order == "linear":
            self.unnormalized_lambdas = nn.Parameter(torch.rand(self.input_dim, self.count_bins))
            
    def forward(self, x):
        """x -> u"""
        y, log_detJ = self.spline_op(x)
        self._cache_log_detJ = log_detJ
        return y, log_detJ

    def inverse(self, y):
        """u > x"""
        x, log_detJ = self.spline_op(y, inverse=True)
        self._cache_log_detJ = -log_detJ
        return x, log_detJ

    def spline_op(self, x, **kwargs):
        """Fit N separate splines for each dimension of input"""
        # widths, unnormalized_widths ~ (input_dim, num_bins)
        w = F.softmax(self.unnormalized_widths, dim=-1)
        h = F.softmax(self.unnormalized_heights, dim=-1)
        d = F.softplus(self.unnormalized_derivatives)
        if self.order == 'linear':
            l = torch.sigmoid(self.unnormalized_lambdas)
        else:
            l = None
        y, log_detJ = _monotonic_rational_spline(x, w, h, d, l, bound=self.bound, **kwargs)
        return y, log_detJ