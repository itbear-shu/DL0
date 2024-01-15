import numpy as np
import weakref
import DL0.functions as F

from DL0.core import Variable, as_variable
import DL0.utils as utils


class Parameter(Variable):
    pass


class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(x) for x in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def clear_grads(self):
        for name in self.params():
            name.clear_grad()


class Linear(Layer):

    def __init__(self, out_size, in_size=None, bias=True, dtype=np.float32):
        super(Linear, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_size is not None:
            self.init_W()

        if not bias:
            self.b = None
        else:
            self.b = Parameter(np.random.randn(1), name='b')

    def init_W(self):
        I, O = self.in_size, self.out_size
        W = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)  # 初始化参数
        self.W.data = W

    def forward(self, inputs):
        if self.W.data is None:
            if inputs.ndim <= 1:  # 处理标量和一维向量
                inputs = inputs.reshape(1, -1)
            self.in_size = inputs.shape[1]
            self.init_W()
        return F.linear(inputs, self.W, self.b)


class TwoLayersNet(Layer):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.layer1 = Linear(hidden_size)
        self.layer2 = Linear(output_size)

    def forward(self, inputs):
        y1 = self.layer1(inputs)
        y2 = F.sigmoid(y1)
        return self.layer2(y2)


class CBOW(Layer):
    def __init__(self, in_size, hidden_size, out_size, dtype=np.float32, onehot=False):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.dtype = dtype
        self.onehot = onehot

        # 初始化参数
        self.W_in = Variable(
            np.random.randn(self.in_size, self.hidden_size).astype(self.dtype) * np.sqrt(1 / self.in_size))

        if not self.onehot:
            self.embedding = Embedding(self.W_in)
        else:
            self.embedding = None

        self.W_out = Variable(
            np.random.randn(self.hidden_size, self.out_size).astype(self.dtype) * np.sqrt(1 / self.hidden_size))

    def forward(self, inputs):
        if self.embedding is None:
            A1 = F.matmul(inputs[0][0].reshape(1, -1), self.W_in)
            A2 = F.matmul(inputs[0][1].reshape(1, -1), self.W_in)
            B = (A1 + A2) / 2
        else:
            A = self.embedding(inputs).sum(axis=1)
            B = A / 2
        y = F.matmul(B, self.W_out)
        return y


class Embedding(Layer):
    def __init__(self, W):
        super().__init__()
        self.W = W

    def forward(self, idx):
        return F.embedding(self.W, idx)
