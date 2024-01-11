import numpy as np
from core import Variable

if __name__ == '__main__':
    # with no_grad():
    #     x = Variable(np.ones((100, 100, 100)) * 2)
    #     y = square(square(square(x)))
    #     # y.backward()
    #     print(y)
    x0 = Variable(np.array([3.0, 3.0]))
    x1 = Variable(np.array([2.0, 2.0]))
    x2 = Variable(np.array(5.0))
    y = x0 / x1
    y.backward()
    print(y)
    print(x0.grad)
    print(x1.grad)
    print(y.ndim)
    print(y.dtype)
