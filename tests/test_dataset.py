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
        train_set = DL0.datasets.Spiral()
        print(train_set[0])
        print(len(train_set))

        batch_index = [1, 2, 3, 5, 0, 21]
        batch = [train_set[i] for i in batch_index]
        x = Variable(np.array([example[0] for example in batch]))
        y = Variable(np.array([example[1] for example in batch]))
        print(x.shape)
        print(y.shape)

    def test2(self):
        epochs = 300
        batch_size = 30
        hidden_size = 10
        lr = 0.1

        transforms = DL0.transforms.Compose(
            [DL0.transforms.AsType(np.float64), DL0.transforms.Normalize(mean=1, std=2)])
        # train_set = DL0.datasets.Spiral(transform=DL0.transforms.Normalize())
        train_set = DL0.datasets.Spiral(transform=transforms)
        model = MLP(hidden_size, 3)
        optimizer = SGD(lr).setup(model)

        data_size = len(train_set)
        max_iter = int(np.ceil(data_size / batch_size))

        for epoch in range(epochs):
            # 将下标随机打散
            index = np.random.permutation(data_size)
            sum_loss = 0.

            for i in range(max_iter):
                # 取一个batch
                batch_index = index[i * batch_size:(i + 1) * batch_size]
                batch = [train_set[i] for i in batch_index]
                x = Variable(np.array([example[0] for example in batch]))
                y = Variable(np.array([example[1] for example in batch]))

                loss = F.softmax_cross_entropy_error(model(x), y)
                model.clear_grads()
                loss.backward()
                optimizer.update()

                sum_loss += float(loss.data) * len(batch_index)

            print(f'epoch: {epoch + 1}, avg_loss = {sum_loss / data_size}')

    def test3(self):
        epochs = 300
        batch_size = 30
        hidden_size = 10
        lr = 0.1

        transforms = DL0.transforms.Compose(
            [DL0.transforms.AsType(np.float64), DL0.transforms.Normalize(mean=1, std=2)])
        train_set = DL0.datasets.Spiral(transform=transforms)
        train_loader = DataLoader(train_set, batch_size)
        model = MLP(hidden_size, 3)
        optimizer = SGD(lr).setup(model)

        data_size = len(train_set)

        for epoch in range(epochs):
            sum_loss = 0.

            for x, y in train_loader:
                loss = F.softmax_cross_entropy_error(model(x), y)
                model.clear_grads()
                loss.backward()
                optimizer.update()

                sum_loss += float(loss.data) * len(x)

            print(f'epoch: {epoch + 1}, avg_loss = {sum_loss / data_size}')

    def test4(self):
        x = Variable(np.random.randn(10, 4))
        # x = Variable(np.array([1, 2]))
        # x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y_hat = F.softmax(x)
        y = Variable(np.random.randint(0, 4, 10))
        print(accuracy(y, y_hat))

    def test5(self):
        epochs = 2000
        batch_size = 30
        hidden_size = [50, 40, 30, 20]
        lr = 100

        transforms = DL0.transforms.Compose(
            [DL0.transforms.AsType(np.float64), DL0.transforms.Normalize(mean=0, std=1)])
        train_set = DL0.datasets.Spiral(transform=transforms)
        test_set = DL0.datasets.Spiral(train=False)
        train_loader = DataLoader(train_set, batch_size)
        test_loader = DataLoader(test_set, batch_size, shuffle=False)
        model = MLP(*hidden_size, 9, activation=F.sigmoid)
        # model = MLP(*hidden_size, 9, activation=F.relu)
        # optimizer = SGD(lr).setup(model)
        optimizer = Adam().setup(model)

        train_size = len(train_set) / batch_size
        test_size = len(test_set) / batch_size

        for epoch in range(epochs):
            sum_loss = 0.
            sum_acc = 0.
            for x, y in train_loader:
                y_hat = model(x)
                loss = F.softmax_cross_entropy_error(y_hat, y)
                y_hat = F.softmax(y_hat)
                acc = DL0.utils.accuracy(y, y_hat)
                model.clear_grads()
                loss.backward()
                optimizer.update()

                sum_loss += float(loss.data) * len(x)
                sum_acc += float(acc.data) * len(x)

            test_loss = 0.
            test_acc = 0.
            with DL0.no_grad():
                for x, y in test_loader:
                    y_hat = model(x)
                    loss = F.softmax_cross_entropy_error(y_hat, y)
                    y_hat = F.softmax(y_hat)
                    acc = DL0.utils.accuracy(y, y_hat)

                    test_loss += float(loss.data) * len(x)
                    test_acc += float(acc.data) * len(x)

            print('epoch[{}], train_loss = {:.4f}, train_acc = {:.4f} || test_loss = {:.4f}, test_acc = {:.4f}'.format(
                epoch + 1, sum_loss / train_size, sum_acc / train_size, test_loss / test_size, test_acc / test_size))
