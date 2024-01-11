import numpy as np
from Variable import Variable

'''将标量转为一维向量'''
def as_ndarray(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(as_ndarray(x.data - eps))
    x1 = Variable(as_ndarray(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)