from Variable import Variable
from utils import as_ndarray
import weakref

class Config:
    enable_backprop = True

class Function:
    def __call__(self, *inputs): # inputs: list
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # *xs, 对xs进行解包, [x0, x1] ==> x0, x1
        if not isinstance(ys, tuple): # ys不是tuple，说明返回值只有一个
            ys = (ys, )
        outputs = [Variable(as_ndarray(y)) for y in ys] # 将y转为向量

        if Config.enable_backprop: # 需要进行反向传播
            self.generation = max([x.generation for x in inputs]) # 函数的辈分等于输入的辈分中的最大值
            for output in outputs:
                output.set_creator(self) # 记录output的创建者

        self.inputs = inputs # 记录输入值，方便backward()

        self.outputs = [weakref.ref(output) for output in outputs] # 弱引用, 使用output()访问实际数据

        return outputs if len(outputs) > 1 else outputs[0] # 返回一个或多个

    def forward(self, *xs): # forward()接口
        raise NotImplementedError()

    def backward(self, *gys): # backward()接口
        raise NotImplementedError()