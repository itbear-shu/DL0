import unittest
from DL0 import Variable
import numpy as np

from DL0.core import numerical_diff
from DL0.math import square


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.))
        y = square(x)
        y.backward()
        expected = np.array(6.)
        self.assertEqual(x.grad.data, expected)

    def test_grad(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        numerical_grad = numerical_diff(square, x)
        print(x.grad, numerical_grad)
        self.assertAlmostEqual(x.grad.data, numerical_grad)

    def test_Sphere(self):
        x = Variable(np.array(1.))
        y = Variable(np.array(1.))
        z = x ** 2 + y ** 2
        z.backward()
        self.assertEqual(x.grad.data, 2.0)
        self.assertEqual(y.grad.data, 2.0)

    def test_matyas(self):
        def matyas(x, y):
            return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

        x = Variable(np.array(1.))
        y = Variable(np.array(1.))
        z = matyas(x, y)
        z.backward()
        self.assertAlmostEqual(x.grad.data, 0.04)
        self.assertAlmostEqual(y.grad.data, 0.04)
