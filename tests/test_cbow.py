import numpy as np
from DL0.core import Variable
from DL0.layers import CBOW
import DL0.functions as F
import unittest
import DL0.utils as utils
from DL0.datasets import CBOWDataset
from DL0.dataloader import DataLoader
from DL0.optimizers import SGD, Adam


class TestCBOW(unittest.TestCase):
    def test1(self):
        x = Variable(np.array([[1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]))
        net = CBOW(7, 3, 7)
        print(net(x).data)

    def test2(self):
        # x = np.array([[0, 2]])
        # print(utils.convert_to_onehot(x, 3))
        text = 'You say good and I say hello.'
        corpus, word_to_id, id_to_word = utils.preprocess(text)
        contexts, target = utils.create_contexts_target(corpus)
        vocab_size = len(word_to_id)
        contexts, target = utils.convert_to_onehot(contexts, vocab_size), utils.convert_to_onehot(target, vocab_size)
        print(contexts)
        print(target)

    def test3(self):
        onehot = False
        train_dataset = CBOWDataset(to_onehot=onehot)
        test_dataset = CBOWDataset(train=False)
        batch_size = 1
        train_loader = DataLoader(train_dataset, batch_size)
        test_loader = DataLoader(test_dataset, batch_size)
        epochs = 100
        lr = 1

        model = CBOW(train_dataset.vocab_size, 3, train_dataset.vocab_size, onehot=onehot)
        optimizer = SGD(lr).setup(model)
        # optimizer = Adam().setup(model)

        for i in range(epochs):
            train_loss = 0.

            for x, y in train_loader:
                y_hat = model(x)
                loss = F.softmax_cross_entropy_error(y_hat, y)
                model.clear_grads()
                loss.backward()
                optimizer.update()

                train_loss += float(loss.data) * len(x)

            print('epoch {}: train_loss = {}'.format(i + 1, train_loss / train_dataset.vocab_size))



