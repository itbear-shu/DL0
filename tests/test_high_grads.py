import unittest
from DL0 import Variable
import numpy as np
import DL0.math as math


class SquareTest(unittest.TestCase):
    def test_f1(self):
        x = Variable(np.array(2.0))
        y = x ** 4 - 2 * x ** 2
        y.backward(create_graph=True)
        self.assertEqual(x.grad.data, 24.0)

        gx = x.grad
        x.clear_grad()
        gx.backward()
        self.assertEqual(x.grad.data, 44.0)

    def test_sin(self):
        x = Variable(np.array(1.))
        y = math.sin(x)
        y.backward(create_graph=True)

        for i in range(3):
            gx = x.grad
            x.clear_grad()
            gx.backward(create_graph=True)
            print(x.grad)

    def test_tanh(self):
        x = Variable(np.array(0.))
        y = math.tanh(x)
        y.backward()
        self.assertEqual(x.grad.data, 1.)