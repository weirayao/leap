import numpy as np 

def amari_distance(m_pred, m_true):
    """Calculates normalized amari distance [0,1] between two matrices"""
    r = np.linalg.inv(m_pred) @ m_true
    p = r.shape[0]
    abs_r = np.abs(r)
    l = np.sum(np.sum(abs_r, axis=1, keepdims=True) / np.max(abs_r, axis=1, keepdims=True) - 1, axis=0)
    r = np.sum(np.sum(abs_r, axis=0, keepdims=True) / np.max(abs_r, axis=0, keepdims=True) - 1, axis=1)
    return 1/(2*p*(p-1)) * (l+r)