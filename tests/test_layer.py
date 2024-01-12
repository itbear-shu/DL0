import unittest
import numpy as np
from DL0.core import Variable
from DL0.layers import Parameter
import DL0.functions as F
import DL0.layers as L
import DL0.models as M


class TestLayer(unittest.TestCase):
    def test1(self):
        x0 = Parameter(np.array(1.))
        x1 = Parameter(np.array(1.))
        x2 = Variable(np.array(1.))
        layer = L.Layer()
        layer.x0 = x0
        layer.x1 = x1
        layer.x2 = x2
        print(layer._params)

        for name in layer._params:
            print(layer.__dict__[name])

    def test2(self):
        # 生成数据
        X = Variable(np.random.randn(1000, 4)) + 10
        y = F.sin(X)

        epochs = 1000
        lr = 1e-2

        L1 = L.Linear(3)
        L2 = L.Linear(4)

        def predict(x):
            A = L1(x)
            B = F.sigmoid(A)
            C = L2(B)
            return C

        for i in range(epochs):
            y_hat = predict(X)
            loss = F.mean_squared_error(y, y_hat)
            L1.clear_grads()
            L2.clear_grads()
            loss.backward()

            for l in [L1, L2]:
                for param in l.params():
                    param.data -= lr * param.grad.data

            print(f'epoch {i + 1}: loss = {loss.data}')
        print(f'W1 = {L1.W.data}')
        print(f'b1 = {L1.b.data}')
        print(f'W2 = {L2.W.data}')
        print(f'b2 = {L2.b.data}')

    def test3(self):
        # 生成数据
        X = Variable(np.random.randn(1000, 4)) + 10
        y = F.sin(X)

        epochs = 1000
        lr = 1e-2

        # model = L.TwoLayersNet(5, 4)
        model = M.TwoLayerModel(5, 4)

        for i in range(epochs):
            y_hat = model(X)
            loss = F.mean_squared_error(y, y_hat)
            model.clear_grads()
            loss.backward()

            for param in model.params():
                param.data -= lr * param.grad.data

            print(f'epoch {i + 1}: loss = {loss.data}')
        # print(f'W1 = {model.layer1.W.data}')
        # print(f'b1 = {model.layer1.b.data}')
        # print(f'W2 = {model.layer2.W.data}')
        # print(f'b2 = {model.layer2.b.data}')
        print(f'W1 = {model.layer.layer1.W.data}')
        print(f'b1 = {model.layer.layer1.b.data}')
        print(f'W2 = {model.layer.layer2.W.data}')
        print(f'b2 = {model.layer.layer2.b.data}')

    def test_MLP(self):
        # 生成数据
        X = Variable(np.random.randn(1000, 4)) + 10
        y = F.sin(X)

        epochs = 1000
        lr = 1e-2

        model = M.MLP((4, 2, 3, 4))

        # for i in range(epochs):
        #     y_hat = model(X)
        #     loss = F.mean_squared_error(y, y_hat)
        #     model.clear_grads()
        #     loss.backward()
        #
        #     for param in model.params():
        #         param.data -= lr * param.grad.data
        #
        #     print(f'epoch {i + 1}: loss = {loss.data}')
        # print(f'W1 = {model.layers[0].W.data}')
        # print(f'b1 = {model.layers[0].b.data}')
        # print(f'W2 = {model.layers[1].W.data}')
        # print(f'b2 = {model.layers[1].b.data}')
