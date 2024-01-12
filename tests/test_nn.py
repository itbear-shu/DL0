import numpy as np
from DL0.core import Variable
import DL0.functions as F
import unittest


class TestNN(unittest.TestCase):
    def test_reshape(self):
        x = Variable(np.array([[3, 3, 3], [2, 2, 2]]))
        y = F.reshape(x, (3, 2))
        y.backward()
        # self.assertEqual(y, 1)
        self.assertEqual(x.grad, 1)

    def test_reshape2(self):
        x = Variable(np.random.randn(2, 2, 3))
        y = x.reshape(4, 3)
        y.backward()
        self.assertEqual(x.grad, 1)

    def test_transpose(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.transpose(x, (1, 0))
        y.backward()
        # self.assertEqual(y, 1)
        self.assertEqual(x.grad, 1)

    def test_transpose2(self):
        x = Variable(np.random.randn(2, 3))
        y = x.T
        y.backward()
        self.assertEqual(x.grad, 1)

    def test_sum(self):
        x = Variable(np.array([[1, 2], [3, 4], [5, 6]]))
        y = F.sum(x, axis=0, keepdims=True)
        y.backward()
        # self.assertEqual(y, 1)
        print(x.grad)

    def test_broadcast(self):
        x = Variable(np.array([1, 2]))
        y = F.broadcast_to(x, (3, 2))
        # print(y)
        y.backward()
        print(x.grad)

    def test_add(self):
        x = Variable(np.array([1, 2, 3]))
        y = Variable(np.array([10]))
        z = x / y
        z.backward()
        print(z)
        print(x.grad)
        print(y.grad)