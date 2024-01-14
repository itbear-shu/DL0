import unittest

import numpy as np

from DL0.core import Variable
import DL0.functions as F
import DL0.layers as L


class TestSoftmax(unittest.TestCase):
    def test1(self):
        x = Variable(np.random.randn(2, 4))
        # x = Variable(np.array([1, 2]))
        # x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.softmax(F.sin(L.Linear(4)(x + 1)))
        y.backward()
        # self.assertEqual(F.sum(y).data, 1)
        print(y.data)
        print(x.grad.data)
