import unittest
import numpy
import numpy as np

from DL0.core import Variable
import DL0.functions as F


class TestActivation(unittest.TestCase):
    def test_relu(self):
        x = Variable(np.random.randn(4, 5))
        print(x.data)
        y = F.relu(x)
        y.backward()
        print(y.data)
        print(x.grad.data)