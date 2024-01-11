from Variable import Variable
from utils import as_ndarray

class Function:
    def __call__(self, input_: Variable):
        x = input_.data
        y = self.forward(x)
        output = Variable(as_ndarray(y)) # 将y转为向量
        output.set_creator(self) # 记录output的创建者
        self.input_ = input_ # 记录输入值，方便backward()
        self.output = output
        return output

    def forward(self, x): # forward()接口
        raise NotImplementedError()

    def backward(self, gy): # backward()接口
        raise NotImplementedError()