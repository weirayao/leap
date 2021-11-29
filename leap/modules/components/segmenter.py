import torch
import torch.nn as nn
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class KMeansObjectifier(nn.Module):
    """Converts RGB into object channels by K-Means"""
    def __init__(self):
        with open("/home/cmu_wyao/kmeans_segmenter.pkl", "wb") as f:
            self.segmenter = pickle.load(f)
    
    def forward(self, x):
        with torch.no_grad

class ConnCompObjectifier(nn.Module):
    """Object/instance segmentation by connected components algorithm
    and Hungarian matching
    """
    def __init__(self, width, height):
        raise NotImplementedError

class MaskRCNN(nn.Module):
    def __init__(self, width, height):
        raise NotImplementedError