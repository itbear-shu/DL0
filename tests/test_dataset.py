import unittest

import numpy as np

from DL0.core import Variable
import DL0.functions as F
from DL0.models import MLP
from DL0.optimizers import SGD
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

        transforms = DL0.transforms.Compose([DL0.transforms.AsType(np.float64), DL0.transforms.Normalize(mean=1, std=2)])
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

            print(f'epoch: {epoch + 1}, avg_loss = {sum_loss/data_size}')
