import unittest
import numpy as np
from DL0.core import Variable
import DL0.functions as F
from DL0.models import MLP
from DL0.optimizers import SGD


class TestOptimizers(unittest.TestCase):
    def test1(self):
        X = Variable(np.random.randn(100, 5))
        y = F.cos(X)

        model = MLP(6, 5)
        lr = 1e-2
        optimizer = SGD(lr).setup(model)
        epochs = 10

        for i in range(epochs):
            y_hat = model(X)
            loss = F.mean_squared_error(y, y_hat)
            model.clear_grads()
            loss.backward()
            optimizer.update()

            print(i + 1, loss.data)


