import numpy as np
import weakref
import contextlib


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    """判断上下文的函数"""
    # 预处理
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        # 后处理
        setattr(Config, name, old_value)


def no_grad():
    """不计算梯度"""
    """使用：
            with no_grad():
                ...
    """
    return using_config('enable_backprop', False)


def as_ndarray(x):
    """将标量转为一维向量"""
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(x):
    if not isinstance(x, Variable):
        return Variable(x)
    return x


class Function:
    def __call__(self, *inputs):  # inputs: list
        inputs = [as_variable(input_) for input_ in inputs] # 把其他类型均转为Variable类型

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # *xs, 对xs进行解包, [x0, x1] ==> x0, x1
        if not isinstance(ys, tuple):  # ys不是tuple，说明返回值只有一个
            ys = (ys,)
        outputs = [Variable(as_ndarray(y)) for y in ys]  # 将y转为向量

        if Config.enable_backprop:  # 需要进行反向传播
            self.generation = max([x.generation for x in inputs])  # 函数的辈分等于输入的辈分中的最大值
            for output in outputs:
                output.set_creator(self)  # 记录output的创建者

        self.inputs = inputs  # 记录输入值，方便backward()

        self.outputs = [weakref.ref(output) for output in outputs]  # 弱引用, 使用output()访问实际数据

        return outputs if len(outputs) > 1 else outputs[0]  # 返回一个或多个

    def forward(self, *xs):  # forward()接口
        raise NotImplementedError()

    def backward(self, *gys):  # backward()接口
        raise NotImplementedError()


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
        self.creator = func
        self.generation = func.generation + 1  # 变量的辈分函数的辈分+1

    '''
    def backward(self): # 递归实现
        f = self.creator # 获取函数创建者
        if f is None: return
        x = f.input_ # 获取函数的输入
        x.grad = f.backward(self.grad) # 调用函数f的backward()
        x.backward() # 递归调用，直至creator为空
    '''

    def backward(self, retain_grad=False):  # 循环实现
        # 为了省略y.grad = np.array(1.0)
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        fs = []
        seen_set = set()  # 记录所有已记录的函数

        def add_func(f):
            if f not in seen_set:
                fs.append(f)
                seen_set.add(f)
                fs.sort(key=lambda x: x.generation)  # 函数按辈分从小到大排序

        add_func(self.creator)

        while fs:
            f = fs.pop()  # 去除辈分最大的函数
            gys = [output().grad for output in f.outputs]  # 多输出值, output是弱引用
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)  # 单输出值转为tuple
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad += gx
                if x.creator is not None:
                    add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # 只保留最终结果的导数

    def clear_grad(self):
        self.grad = None

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

    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __truediv__(self, other):
        """真除法"""
        return div(self, other)


class Add(Function):
    """y = x0 + x1"""

    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gy):
        # dx0 = 1, dx1 = 1
        return 1 * gy, 1 * gy


def add(x0, x1):
    x1 = as_ndarray(x1)
    return Add()(x0, x1)


class Sub(Function):
    """y = x0 - x1"""

    def forward(self, x0, x1):
        return x0 - x1

    def backward(self, gy):
        return gy, gy


def sub(x0, x1):
    x1 = as_ndarray(x1)
    return Sub()(x0, x1)


class Mul(Function):
    """y = x0 * x1"""

    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy * x1
        gx1 = gy * x0
        return gx0, gx1


def mul(x0, x1):
    x1 = as_variable(x1)
    return Mul()(x0, x1)


def numerical_diff(f, x, eps=1e-4):
    """数值微分"""
    x0 = Variable(as_ndarray(x.data - eps))
    x1 = Variable(as_ndarray(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class Div(Function):
    """ y = x0 / x1"""

    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = - gy * x0 / (x1 ** 2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_ndarray(x1)
    return Div()(x0, x1)
