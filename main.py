import numpy as np
from Variable import Variable
from functions import square, exp

if __name__ == '__main__':
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()
    print(x.grad)
