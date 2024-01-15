from DL0.core import Function, as_variable, as_ndarray
import numpy as np
import DL0.utils as utils


########################################################
# 基础数学函数
########################################################

class Square(Function):
    """y = x^2"""

    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        # dx = 2 * x
        x = self.inputs[0]
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
        # dx = e^x
        x = self.inputs[0]
        gx = exp(x) * gy
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


########################################################
# 张量特性
########################################################

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


class Squeeze(Function):
    def __init__(self, axis):
        self.axis = axis
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        return np.squeeze(x, axis=self.axis)

    def backward(self, gy):
        x, = self.inputs
        return gy * x.reshape(self.x_shape)


def squeeze(x, axis):
    return Squeeze(axis)(x)


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


########################################################
# 激活函数
########################################################

class Sigmoid(Function):
    """y = 1 / (1 + e^-x)"""

    def forward(self, x):
        eps = 1e-5
        return 1 / (1 + np.exp(-x) + eps)

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)


class ReLU(Function):
    def __init__(self):
        self.eps = 1e-9

    def forward(self, x):
        return np.maximum(x, 0.) + self.eps

    def backward(self, gy):
        x = self.inputs[0]
        return gy * (x.data > 0) + self.eps


def relu(x):
    return ReLU()(x)


class Softmax(Function):
    def __init__(self):
        self.axis = 0
        self.eps = 1e-9

    def forward(self, x):
        y = x - x.max(axis=self.axis, keepdims=True)
        exp_x = np.exp(y)
        if exp_x.ndim > 1:
            self.axis = 1
        return exp_x / (np.sum(exp_x, axis=self.axis, keepdims=True) + self.eps)

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sum_dx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sum_dx
        return gx


def softmax(x):
    return Softmax()(x)


########################################################
# 线性函数
########################################################

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
        gb = None if b.data is None else sum_to(gy, b.shape)
        return gx, gw, gb


def linear(X, W, b):
    return Linear()(X, W, b)


########################################################
# 损失函数
# y：真实label
# y_hat：预测值
########################################################

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


class CrossEntropy(Function):
    def __init__(self):
        self.eps = 1e-9

    def forward(self, y, y_hat):
        return -np.sum(y * np.log(y_hat + self.eps)) / len(y)

    def backward(self, gy):
        y, y_hat = self.inputs
        return - gy * y / (y_hat + self.eps) / len(y)


def cross_entropy_error(y, y_hat):
    return CrossEntropy()(y, y_hat)


class SoftmaxCrossEntropy(Function):
    def __init__(self):
        self.axis = 0
        self.eps = 1e-9
        self.y_hat = None
        self.y = None

    def forward(self, x, y):
        if y.ndim <= 1:
            y = np.eye(x.shape[1], dtype=y.dtype)[y.data]
        self.y = y
        exp_x = np.exp(x - x.max(axis=self.axis, keepdims=True))
        if exp_x.ndim > 1:
            self.axis = 1
        y_hat = exp_x / (np.sum(exp_x, axis=self.axis, keepdims=True) + self.eps)
        self.y_hat = as_variable(y_hat)
        return -np.sum(y * np.log(y_hat + self.eps)) / len(y)

    def backward(self, gy):
        return gy * (self.y_hat - self.y) / len(self.y)


def softmax_cross_entropy_error(x, y):
    return SoftmaxCrossEntropy()(x, y)


########################################################
# NLP
########################################################
class Embedding(Function):
    def forward(self, W, idx):
        return W[idx]

    def backward(self, gy):
        W, idx = self.inputs
        # for i, word_id in enumerate(idx.data):
        #     W.data[word_id] += gy.data[i]
        np.add.at(W.data, idx.data, gy.data)
        return W


def embedding(W, idx):
    return Embedding()(W, idx)
