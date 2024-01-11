from Function import Function
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

class Add(Function):
    """y = x0 + x1"""
    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gy):
        # dx0 = 1, dx1 = 1
        return 1 * gy, 1 * gy

def add(x0, x1):
    f = Add()
    return f(x0, x1)