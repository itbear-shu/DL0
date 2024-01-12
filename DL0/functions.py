from DL0.core import Function, as_variable
import numpy as np
import DL0.utils as utils


class Square(Function):
    """y = x^2"""

    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        # dy = 2 * x
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    f = Square()
    return f(x)


class Exp(Function):
    """y = e^x"""

    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        # dy = e^x
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


def exp(x):
    f = Exp()
    return f(x)


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        x = self.inputs[0]
        gx = cos(x) * gy
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        return np.cos(x)

    def backward(self, gy):
        x = self.inputs[0]
        gx = -sin(x) * gy
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    """y = (e^x - e^-x) / (e^x + e^-x)"""

    def forward(self, x):
        return np.tanh(x)

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y ** 2)
        return gx


def tanh(x):
    return Tanh()(x)


class Reshape(Function):
    def __init__(self, shape):
        self.target_shape = shape
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.target_shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes):
        self.target_axes = axes

    def forward(self, x):
        return np.transpose(x, self.target_axes)

    def backward(self, gy):
        if self.target_axes is None:
            return transpose(gy)
        axes_len = len(self.target_axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.target_axes]))
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    if not isinstance(axes, (tuple, list)):
        axes = None
    return Transpose(axes)(x)


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        return np.sum(x, axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        return broadcast_to(gy, self.x_shape)


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTO(Function):
    def __init__(self, shape):
        self.target_shape = shape
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        gx = np.broadcast_to(x, self.target_shape)
        return gx

    def backward(self, gy):
        return sum_to(gy, self.x_shape)


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTO(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.target_shape = shape
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        gx = utils.sum_to(x, self.target_shape)
        return gx

    def backward(self, gy):
        return broadcast_to(gy, self.x_shape)


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class MatMul(Function):
    def forward(self, X, W):
        return np.dot(X, W)

    def backward(self, gy):
        X, W = self.inputs
        gx = matmul(gy, W.T)
        gw = matmul(X.T, gy)
        return gx, gw


def matmul(X, W):
    return MatMul()(X, W)


class Linear(Function):
    def forward(self, X, W, b=None):
        t = np.dot(X, W)
        if b is None:
            return t
        return t + b

    def backward(self, gy):
        X, W, b = self.inputs
        gx = matmul(gy, W.T)
        gw = matmul(X.T, gy)
        gb = None if b is None else sum_to(gy, b.shape)
        return gx, gw, gb


def linear(X, W, b):
    return Linear()(X, W, b)


class MSE(Function):
    """Mean Squared Error: 均方差"""

    def forward(self, y, y_hat):
        diff = y - y_hat
        return np.sum(diff ** 2) / len(y)

    def backward(self, gy):
        y, y_hat = self.inputs
        gx0 = gy * 2. * (y - y_hat) / len(y)
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(y, y_hat):
    return MSE()(y, y_hat)
