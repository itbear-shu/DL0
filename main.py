import DL0
from DL0.dataloader import DataLoader
from DL0.models import MLP
from DL0.optimizers import Adam
import DL0.functions as F
import numpy as np


def test5():
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


if __name__ == '__main__':
    test5()
