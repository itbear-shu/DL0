import numpy as np
import DL0.utils as utils
import os


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


class CBOWDataset(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, to_onehot=False):
        self.to_onehot = to_onehot
        self.vocab_size = None
        super().__init__(train=train, transform=transform, target_transform=target_transform)

    def prepare(self):
        if self.train:
            path = os.path.dirname(__file__) + '/data/cbow/train.txt'
        else:
            path = os.path.dirname(__file__) + '/data/cbow/test.txt'
        text = ''
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                text += line
        corpus, word_to_id, id_to_word = utils.preprocess(text)
        contexts, target = utils.create_contexts_target(corpus)
        self.vocab_size = len(word_to_id)
        if not self.to_onehot:
            self.data, self.label = contexts, utils.convert_to_onehot(target, self.vocab_size)
        else:
            self.data, self.label = utils.convert_to_onehot(contexts, self.vocab_size), \
                                    utils.convert_to_onehot(target, self.vocab_size)
