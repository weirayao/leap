import math
import numpy as np

def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    Q = I - unit/n
    
    return np.dot(np.dot(Q, K), Q)

def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / sigma / sigma
    np.exp(KX, KX)
    return KX

def HSIC(X, Y):
    return np.sum(centering(rbf(X))*centering(rbf(Y)))

if __name__ == '__main__':
    X = np.random.randn(10, 5)
    Y = np.random.randn(10, 5)
    print HSIC(X, Y)
    print HSIC(X, X)