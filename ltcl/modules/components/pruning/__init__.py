'''This folder and scripts are copies from LASSONet and scikit-learn

LassoNet: Neural Networks with Feature Sparsity
https://github.com/lasso-net/lassonet

Mutual Information regression
https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html

'''
from .model import LassoNet
from .prox import prox
from .interfaces import LassoNetClassifier, LassoNetRegressor, lassonet_path
from .utils import plot_path
from sklearn.feature_selection import mutual_info_regression