import numpy as np
import DL0.utils as utils


class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:  # 给x转换
            self.transform = lambda x: x
        if self.target_transform is None:  # 给label转换
            self.target_transform = lambda x: x
        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)  # 如果index不是标量，则退出
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), self.target_transform(self.label[index])

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass


class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = utils.get_spiral(self.train)
