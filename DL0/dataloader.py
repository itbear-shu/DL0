import numpy as np
import math
from DL0.core import Variable


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.iteration = 0
        self.index = None

        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:  # 训练集
            self.index = np.random.permutation(self.data_size)
        else:
            self.index = np.arange(self.data_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        batch_index = self.index[self.iteration * self.batch_size:(self.iteration + 1) * self.batch_size]
        batch = [self.dataset[i] for i in batch_index]
        x = Variable(np.array([example[0] for example in batch]))
        y = Variable(np.array([example[1] for example in batch]))

        self.iteration += 1
        return x, y

    def next(self):
        return self.__next__()
