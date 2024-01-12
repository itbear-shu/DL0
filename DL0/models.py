import DL0.layers as L
import DL0.functions as F


class Model(L.Layer):
    pass


class TwoLayerModel(Model):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.layer = L.TwoLayersNet(hidden_size, output_size)

    def forward(self, inputs):
        return self.layer(inputs)


class MLP(Model):
    def __init__(self, sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for index in range(len(sizes)):
            layer = L.Linear(sizes[index])
            self.__setattr__('layer' + str(index), layer)
            self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers[:-1]:  # 倒数第二层
            inputs = self.activation(layer(inputs))
        return self.layers[-1](inputs)  # 最后一层没有激活函数
