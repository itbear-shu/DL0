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
        self.generation = 0 # 设置辈分

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 # 变量的辈分函数的辈分+1

    '''
    def backward(self): # 递归实现
        f = self.creator # 获取函数创建者
        if f is None: return
        x = f.input_ # 获取函数的输入
        x.grad = f.backward(self.grad) # 调用函数f的backward()
        x.backward() # 递归调用，直至creator为空
    '''

    def backward(self, retain_grad=False): # 循环实现
        # 为了省略y.grad = np.array(1.0)
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        fs = []
        seen_set = set() # 记录所有已记录的函数
        def add_func(f):
            if f not in seen_set:
                fs.append(f)
                seen_set.add(f)
                fs.sort(key=lambda x : x.generation) # 函数按辈分从小到大排序
        add_func(self.creator)

        while fs:
            f = fs.pop() # 去除辈分最大的函数
            gys = [output().grad for output in f.outputs] # 多输出值, output是弱引用
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, ) # 单输出值转为tuple
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad += gx
                if x.creator is not None:
                    add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # 只保留最终结果的导数

    def clear_grad(self):
        self.grad = None