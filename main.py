import numpy as np
from Variable import Variable
from functions import square, exp, add
from Function import Config

if __name__ == '__main__':
    Config.enable_backprop = False
    x = Variable(np.ones((100, 100, 100)) * 2)
    y = square(square(square(x)))
    # y.backward()
    print(y.data)
