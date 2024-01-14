import numpy as np
import weakref
import contextlib
import DL0


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
        inputs = [as_variable(input_) for input_ in inputs]  # 把其他类型均转为Variable类型

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # *xs, 对xs进行解包, [x0, x1] ==> x0, x1  (np.array)
        if not isinstance(ys, tuple):  # ys不是tuple，说明返回值只有一个
            ys = (ys,)
        outputs = [Variable(as_ndarray(y)) for y in ys]  # 将y转为向量 (Variable)

        if Config.enable_backprop:  # 需要进行反向传播
            self.generation = max([x.generation for x in inputs])  # 函数的辈分等于输入的辈分中的最大值
            for output in outputs:
                output.set_creator(self)  # 记录output的创建者
            self.inputs = inputs  # 记录输入值，方便backward()
            self.outputs = [weakref.ref(output) for output in outputs]  # 弱引用, 使用output()访问实际数据

        return outputs if len(outputs) > 1 else outputs[0]  # 返回一个或多个

    def forward(self, *xs):  # forward()接口
        """np.ndarray"""
        raise NotImplementedError()

    def backward(self, *gys):  # backward()接口
        """Variable"""
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

    def backward(self, retain_grad=False, create_graph=False):  # 循环实现
        """
            retain_grad: 是否保存中间变量的grad
            create_graph: 是否在反向传播时创建计算图，方便求高阶导数
        """
        # 为了省略y.grad = np.array(1.0)
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        fs = []
        seen_set = set()  # 记录所有已记录的函数

        def add_func(func):
            if func not in seen_set:
                fs.append(func)
                seen_set.add(func)
                fs.sort(key=lambda w: w.generation)  # 函数按辈分从小到大排序

        add_func(self.creator)

        while fs:
            f = fs.pop()  # 取出辈分最大的函数
            gys = [output().grad for output in f.outputs]  # 多输出值, output是弱引用
            with using_config('enable_backprop', create_graph):  # 高阶导，启用反向传播[create_graph=True]
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)  # 单输出值转为tuple
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx  # Variable
                    if x.creator is not None:
                        add_func(x.creator)
                if not retain_grad:
                    for output in f.outputs:
                        output().grad = None  # 只保留最终结果的导数

    def clear_grad(self):
        self.grad = None

    def matmul(self, other):
        return DL0.functions.matmul(self, other)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return DL0.functions.reshape(self, shape)

    def transpose(self, axes=None):
        if not isinstance(axes, (tuple, list)):
            axes = None
        return DL0.functions.transpose(self, axes)

    def sum(self, axis=None, keepdims=False):
        return DL0.functions.sum(self, axis, keepdims)

    @property
    def T(self):
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

    def __abs__(self):
        return abs_(self)


def numerical_diff(f, x, eps=1e-4):
    """数值微分"""
    x0 = Variable(as_ndarray(x.data - eps))
    x1 = Variable(as_ndarray(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


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


class Sub(Function):
    """y = x0 - x1"""

    def __init__(self):
        self.x0_shape = None
        self.x1_shape = None

    def forward(self, x0, x1):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        return x0 - x1

    def backward(self, gy):
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:
            gx0 = DL0.functions.sum_to(gx0, self.x0_shape)
            gx1 = DL0.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1):
    x0 = as_ndarray(x0)
    x1 = as_ndarray(x1)
    return Sub()(x0, x1)


class Mul(Function):
    """y = x0 * x1"""

    def __init__(self):
        self.x0_shape = None
        self.x1_shape = None

    def forward(self, x0, x1):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if self.x0_shape != self.x1_shape:
            gx0 = DL0.functions.sum_to(gx0, self.x0_shape)
            gx1 = DL0.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def mul(x0, x1):
    x0 = as_ndarray(x0)
    x1 = as_ndarray(x1)
    return Mul()(x0, x1)


class Div(Function):
    """ y = x0 / x1"""

    def __init__(self):
        self.x0_shape = None
        self.x1_shape = None

    def forward(self, x0, x1):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = - gy * x0 / (x1 ** 2)
        if self.x0_shape != self.x1_shape:
            gx0 = DL0.functions.sum_to(gx0, self.x0_shape)
            gx1 = DL0.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def div(x0, x1):
    x0 = as_ndarray(x0)
    x1 = as_ndarray(x1)
    return Div()(x0, x1)


class Neg(Function):
    """y = -x"""

    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Pow(Function):
    """y = x ** p"""

    def forward(self, x, p):
        return x ** p

    def backward(self, gy):
        x, p = self.inputs
        return p * x ** (p - 1) * gy


def pow_(x, p):
    p = as_ndarray(p)
    return Pow()(x, p)


class Abs(Function):
    def forward(self, x):
        return np.abs(x)

    def backward(self, gy):
        x = self.inputs[0].data
        yield x.any() == 0
        if x.all() > 0:
            return gy
        else:
            return -gy


def abs_(x):
    return Abs()(x)
