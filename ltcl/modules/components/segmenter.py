import torch
import torch.nn as nn
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class ColorBasedObjectifier(nn.Module):
    """Converts RGB image into binary/encoding object channels"""
    def __init__(self, width, height):
        raise NotImplementedError

class ConnCompObjectifier(nn.Module):
    """Object/instance segmentation by connected components algorithm
    and Hungarian matching
    """
    def __init__(self, width, height):
        raise NotImplementedError

class MaskRCNN(nn.Module):
    def __init__(self, width, height):
        raise NotImplementedError