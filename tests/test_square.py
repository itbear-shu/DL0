import unittest
from DL0 import Variable
import numpy as np
from DL0.math import square
from DL0.utils import numerical_diff

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
        self.assertEqual(x.grad, expected)

    def test_grad(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        numerical_grad = numerical_diff(square, x)
        print(x.grad, numerical_grad)
        self.assertAlmostEqual(x.grad, numerical_grad)