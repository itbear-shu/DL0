# DL0
根据`《深度学习入门：自制框架》`一书手把手从0开始自制深度学习框架。

原书源代码见：`dl-from-scratch`文件

所需包仅为：`numpy`

## 效果

> 可以实现类似`PyTorch`的基础功能：Dataset、DataLoader、transforms、model(layer)、optimizer、loss_function

```python
def test(self):
    epochs = 2000
    batch_size = 30
    hidden_size = [50, 40, 30, 20] # 隐藏层
    lr = 1e-3 # 学习率

    # 数据转换
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
        with DL0.no_grad(): # 不保存每一个算子的outputs
            for x, y in test_loader:
                y_hat = model(x)
                loss = F.softmax_cross_entropy_error(y_hat, y)
                y_hat = F.softmax(y_hat)
                acc = DL0.utils.accuracy(y, y_hat)

                test_loss += float(loss.data) * len(x)
                test_acc += float(acc.data) * len(x)

        print('epoch[{}], train_loss = {:.4f}, train_acc = {:.4f} || test_loss = {:.4f}, test_acc = {:.4f}'.format(
            epoch + 1, sum_loss / train_size, sum_acc / train_size, test_loss / test_size, test_acc / test_size))
```

## Variable

```python
class Variable:
    """变量类"""

    def __init__(self, data: np.ndarray, name=None):
        # 使data只支持np.ndarray
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError(f'{type(data)} is not supported.')
        self.data = data  # 记录数据值
        self.grad = None  # 记录梯度
        self.name = name  # 设置变量名称
        self.creator = None  # 记录创建者
        self.generation = 0  # 设置辈分
		
    def set_creator(self, func):
    	"""记录创建者"""
		self.creator = func
        self.generation = func.generation + 1  # 变量的辈分函数的辈分+1

    def backward(self, retain_grad=False, create_graph=False):  # 循环实现
        """
        	反向传播
            retain_grad: 是否保存中间变量的grad
            create_graph: 是否在反向传播时创建计算图，方便求高阶导数
        """
        pass

    def clear_grad(self): # 清除梯度
        self.grad = None

    def matmul(self, other): # 矩阵相差
        return DL0.functions.matmul(self, other)

    def reshape(self, *shape): # reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return DL0.functions.reshape(self, shape)

    def transpose(self, axes=None): # 矩阵转置
        if not isinstance(axes, (tuple, list)):
            axes = None
        return DL0.functions.transpose(self, axes)

    def sum(self, axis=None, keepdims=False): # sum
        return DL0.functions.sum(self, axis, keepdims)

    @property
    def T(self): # 矩阵转置
        return DL0.functions.transpose(self, None)

    @property
    def shape(self):  # shape
        return self.data.shape

    @property
    def ndim(self):  # 维度
        return self.data.ndim

    @property
    def dtype(self):  # 数据类型
        return self.data.dtype

    @property
    def size(self):  # 元素总数
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'Variable(' + p + ', dtype=' + str(self.dtype) + ')'

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __truediv__(self, other):
        """真除法"""
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __neg__(self):
        """负数"""
        return neg(self)

    def __pow__(self, power, modulo=None):
        """self ** power"""
        return pow_(self, power)
```

## Function

```python
class Function:
    def __call__(self, *inputs):  # inputs: list
        pass

    def forward(self, *xs):  # forward()接口
        """np.ndarray"""
        raise NotImplementedError()

    def backward(self, *gys):  # backward()接口
        """Variable"""
        raise NotImplementedError()
```

## Operators

> 常用算子：加、剪、乘、除、取负、幂函数、平方、指数、sin、cos、tanh、sigmoid、relu、softmax、softmax、sum、broadcast、矩阵相乘

```python
class Add(Function):
    """y = x0 + x1"""

    def __init__(self):
        self.x0_shape = None
        self.x1_shape = None

    def forward(self, x0, x1):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        return x0 + x1

    def backward(self, gy):
        # dx0 = 1, dx1 = 1
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:  # 进行了广播，需要复原
            gx0 = DL0.functions.sum_to(gx0, self.x0_shape)
            gx1 = DL0.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

def add(x0, x1):
    x0 = as_ndarray(x0)
    x1 = as_ndarray(x1)
    return Add()(x0, x1)

class Square(Function):
    """y = x^2"""

    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        # dx = 2 * x
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx

def square(x):
    f = Square()
    return f(x)
```

## Loss function

```python
class MSE(Function):
    """Mean Squared Error: 均方差"""

    def forward(self, y, y_hat):
        diff = y - y_hat
        return np.sum(diff ** 2) / len(y)

    def backward(self, gy):
        y, y_hat = self.inputs
        gx0 = gy * 2. * (y - y_hat) / len(y)
        gx1 = -gx0
        return gx0, gx1

def mean_squared_error(y, y_hat):
    return MSE()(y, y_hat)

class CrossEntropy(Function):
    pass

class SoftmaxCrossEntropy(Function):
	pass
```

## Layer

```python
class Parameter(Variable):
    pass

class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value): # 设置参数名与value的对应
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

    def forward(self, inputs): # 正向传播接口
        raise NotImplementedError()

    def params(self): # 访问Layer的所有参数，返回的是迭代器
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
	pass

class TwoLayersNet(Layer):
    pass
```

## Model

```python
class Model(Layer):
    pass

class TwoLayerModel(Model):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.layer = L.TwoLayersNet(hidden_size, output_size)

    def forward(self, inputs):
        return self.layer(inputs)

class MLP(Model): # 多层感知机模型
    def __init__(self, *sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation # 激活函数
        self.layers = []

        for index in range(len(sizes)):
            layer = L.Linear(sizes[index])
            self.__setattr__('layer' + str(index), layer) # 设置每一层的参数名与参数值的字典
            self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers[:-1]:  # 倒数第二层
            inputs = self.activation(layer(inputs))
        return self.layers[-1](inputs)  # 最后一层没有激活函数
```

## Optimizer

```python
class Optimizer: # 优化器
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]
        for f in self.hooks:
            f(params)

        for param in params:
            self.update_one(param)

    def update_one(self, param): # 更新单个参数接口
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data
        
class MomentumSGD(Optimizer):
    pass

class AdaGrad(Optimizer):
    pass

class AdaDelta(Optimizer):
    pass

class Adam(Optimizer):
    pass
```

## Dataset

```python
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

    def prepare(self): # 数据准备接口
        pass

class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = utils.get_spiral(self.train)
```

## DataLoader

```python
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

    def __iter__(self): # 迭代器
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
```

## Test

见`tests`文件夹
