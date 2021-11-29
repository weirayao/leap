import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle, Polygon

from leap.tools.utils import rand_float, rand_int, calc_dis, norm

import ipdb as pdb

GRAVITY = 9.8
MASS = 1
POSITION

class SpringMassSystem():

    def __init__(self):
        self.n_balls = 3
        self.sparsity = 0.67
        param_load = np.zeros((self.n_ball * (self.n_ball - 1) // 2, 2))
        n_rels = len(param_load)
        num_nonzero = int(n_rels * sparsity)
        choice = np.random.choice(n_rels, size=num_nonzero, replace=False)
        param_load[choice, 0] = 1
        param_load[choice, 1] = np.random.rand(num_nonzero) * 10 + 20
        self.param_load = param_load
        self.radius = 6
        # Init location and speed
        

