from DL0.core import Function
import numpy as np


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
       gx = gy * (1 - y**2)
       return gx

def tanh(x):
    return Tanh()(x)
