import numpy as np

class Variable:
    """变量类"""
    def __init__(self, data: np.ndarray):
        # 使data只支持np.ndarray
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError(f'{type(data)} is not supported.')
        self.data = data # 记录数据值
        self.grad = None # 记录梯度
        self.creator = None # 记录创建者

    def set_creator(self, func):
        self.creator = func

    '''
    def backward(self): # 递归实现
        f = self.creator # 获取函数创建者
        if f is None: return
        x = f.input_ # 获取函数的输入
        x.grad = f.backward(self.grad) # 调用函数f的backward()
        x.backward() # 递归调用，直至creator为空
    '''

    def backward(self): # 循环实现
        # 为了省略y.grad = np.array(1.0)
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        f = self.creator # 获取函数创建者
        while f is not None:
            x, y = f.input_, f.output # 获取函数f的输入值和输出值(Variable类型)
            x.grad = f.backward(y.grad) # 计算x的梯度
            f = x.creator
