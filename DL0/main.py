import numpy as np
from DL0.core import Variable
from DL0.math import sin

def rosenrock(x0, x1):
    return 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2

if __name__ == '__main__':
    # with no_grad():
    #     x = Variable(np.ones((100, 100, 100)) * 2)
    #     y = square(square(square(x)))
    #     # y.backward()
    #     print(y)
    # x = Variable(np.array(np.pi / 4))
    # y = sin(x)
    # y.backward()
    # print(y)
    # print(x.grad)
    x0 = Variable(np.array(0.))
    x1 = Variable(np.array(2.))
    y = rosenrock(x0, x1)
    epochs = 10000
    lr = 1e-4
    for i in range(epochs):
        x0.clear_grad()
        x1.clear_grad()
        y.backward()

        x0.data -= lr * x0.grad
        x1.data -= lr * x1.grad

        if i % 100 == 0:
            print(f'epoch{i + 1}: x0={x0}, x1={x1}')
