import unittest

import numpy as np

from DL0.core import Variable
import DL0.functions as F
import DL0.layers as L


class TestLoss(unittest.TestCase):
    def test1(self):
        x = Variable(np.random.randn(2, 4))
        # x = Variable(np.array([1, 2]))
        # x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.softmax(F.sin(L.Linear(4)(x + 1)))
        y.backward()
        # self.assertEqual(F.sum(y).data, 1)
        print(y.data)
        print(x.grad.data)

    def test2(self):
        x = Variable(np.random.randn(2, 4))
        # x = Variable(np.array([1, 2]))
        # x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y_hat = F.softmax(x)
        print(y_hat.data)
        y = Variable(np.array([[1, 0, 0, 0], [0, 1, 0, 0]]))
        z = F.cross_entropy_error(y, y_hat)
        # z.backward(retain_grad=True)
        # self.assertEqual(F.sum(y).data, 1)
        print(z.data)
        # print(y.grad.data)
        # print(x.grad.data)

    def test3(self):
        x = Variable(np.array([[0.2, 10000, 123123132]]))
        y = Variable(np.array([[1, 0, 0]]))
        loss = F.softmax_cross_entropy_error(x, y)
        print(loss.data)
        loss.backward()
        print(x.grad.data)
