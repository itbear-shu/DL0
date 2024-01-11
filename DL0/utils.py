import numpy as np


def as_ndarray(x):
    """将标量转为一维向量"""
    if np.isscalar(x):
        return np.array(x)
    return x
