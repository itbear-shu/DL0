import DL0 as dl0
import numpy as np

if __name__ == '__main__':
    x1 = dl0.Variable(np.array(4))
    x2 = dl0.Variable(np.array(6))
    y1 = x1 + 2 * x2
    print(y1)
    y1.backward()
    print(x1.grad)
    print(x2.grad)

    x1.clear_grad()
    x2.clear_grad()
    x1 = dl0.Variable(np.array(10))
    x2 = dl0.Variable(np.array(1))
    y2 = x1 / x2 + x1 ** 2 + (-x2)
    print(y2)
    y2.backward()
    print(x1.grad)
    print(x2.grad)
