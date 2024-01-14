import numpy as np
from DL0.core import Variable
import DL0.functions as F


def rosenrock(x0, x1):
    return x0 + x1


if __name__ == '__main__':
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.sin(x) + 2
    print(y)
