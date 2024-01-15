import unittest

import numpy as np

from DL0.core import Variable
import DL0.functions as F
from DL0.models import MLP
from DL0.optimizers import SGD, Adam
from DL0 import DataLoader
from DL0.utils import accuracy

import DL0.datasets


class TestDataset(unittest.TestCase):
    def test1(self):
        x = Variable(np.arange(21).reshape(7, 3))
        print(x.data)
        idx = Variable(np.array([2, 0, 1, 3]))
        y = F.embedding(x, idx)
        print(y)
        y.backward()
        print(x.grad.data)
