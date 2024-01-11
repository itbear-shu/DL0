import numpy as np
from DL0.core import Variable
from DL0.functions import sin


def rosenrock(x0, x1):
    return x0 + x1


if __name__ == '__main__':
    x0 = Variable(np.array(0.))
    x1 = Variable(np.array(2.))
    lr = 1e-4
    y = rosenrock(x0, x1)
    y.backward()
    x3 = x0.grad * lr
    x0.data -= x0.grad.data * lr
    x1.data -= x1.grad.data * lr
