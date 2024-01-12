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

    def test_matmul(self):
        X = Variable(np.arange(12).reshape(3, 4))
        W = Variable(np.arange(16).reshape(4, 4))
        y = X.matmul(W)
        y.backward()
        print(X.shape)
        print(W.shape)

    def test_linear(self):
        # 生成数据
        X = Variable(np.random.rand(100, 4))
        true_W = Variable(np.arange(1, 9).reshape(4, 2))
        true_b = Variable(np.random.rand(1))
        y = F.linear(X, true_W, true_b)
        print('true_W = ', true_W.data)
        print('true_b = ', true_b.data)
        # 初始化参数
        W = Variable(np.random.randn(4, 2))
        b = Variable(np.random.randn(1))
        epochs = 1000
        lr = 1e-2

        for i in range(epochs):
            W.clear_grad()
            b.clear_grad()
            y_hat = F.linear(X, W, b)

            loss = F.mean_squared_error(y, y_hat)
            loss.backward()

            W.data -= lr * W.grad.data
            b.data -= lr * b.grad.data
            print(f'epoch {i + 1}: loss = {loss.data}')
            print(f'W = {W.data}')
            print(f'b = {b.data}')

    def test_sin(self):
        # 生成数据
        X = Variable(np.random.randn(1000, 4)) + 10
        y = F.sin(X)

        epochs = 1000
        lr = 1e-2

        # 初始化参数
        W1 = Variable(np.random.randn(4, 2))
        b1 = Variable(np.random.randn(1))
        W2 = Variable(np.random.randn(2, 4))
        b2 = Variable(np.random.randn(1))

        def net(X):
            A = F.linear(X, W1, b1)
            B = F.sigmoid(A)
            C = F.linear(B, W2, b2)
            return C

        for i in range(epochs):
            y_hat = net(X)
            loss = F.mean_squared_error(y, y_hat)
            W1.clear_grad()
            W2.clear_grad()
            b1.clear_grad()
            W2.clear_grad()
            loss.backward()
            W1.data -= W1.grad.data * lr
            W2.data -= W2.grad.data * lr
            b1.data -= b1.grad.data * lr
            b2.data -= b2.grad.data * lr

            print(f'epoch {i + 1}: loss = {loss.data}')
        print(f'W1 = {W1.data}')
        print(f'b1 = {b1.data}')
        print(f'W2 = {W2.data}')
        print(f'b2 = {b2.data}')