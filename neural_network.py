import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化输入层到隐藏层的权重为随机值
        self.weights_input_to_hidden = np.random.rand(input_size, hidden_size) 
        # 初始化隐藏层到输出层的权重为随机值
        self.weights_hidden_to_output = np.random.rand(hidden_size, output_size)     
        # 初始化隐藏层的偏置为随机值
        self.hidden_bias = np.random.rand(hidden_size)
        # 初始化输出层的偏置为随机值
        self.output_bias = np.random.rand(output_size)

    def sig(self, x):
        # Sigmoid 激活函数
        return 1 / (1 + np.exp(-x))
    
    def sig_de(self, x):
        # Sigmoid 函数的导数，用于反向传播
        return x * (1 - x)

    def forward(self, inputs):
        # 前向传播：从输入层到隐藏层
        self.hi_in = np.dot(inputs, self.weights_input_to_hidden) + self.hidden_bias
        self.hi_ou = self.sig(self.hi_in)

        # 前向传播：从隐藏层到输出层
        self.ou_in = np.dot(self.hi_ou, self.weights_hidden_to_output) + self.output_bias
        self.out = self.sig(self.ou_in)

        return self.out
    
    def backward(self, inputs, ex_ou, learning_rate):
        # 反向传播：计算输出层的误差和梯度
        self.ou_error = ex_ou - self.out
        self.ou_delta = self.ou_error * self.sig_de(self.out)

        # 反向传播：计算隐藏层的误差和梯度
        self.hi_error = np.dot(self.ou_delta, self.weights_hidden_to_output.T)
        self.hi_delta = self.hi_error * self.sig_de(self.hi_ou)

        # 更新隐藏层到输出层的权重
        self.weights_hidden_to_output += np.dot(self.hi_ou.T.reshape(-1, 1), self.ou_delta.reshape(1, -1)) * learning_rate
        # 更新输入层到隐藏层的权重
        self.weights_input_to_hidden += np.dot(inputs.reshape(-1, 1), self.hi_delta.reshape(1, -1)) * learning_rate

        # 更新隐藏层的偏置
        self.hidden_bias += np.sum(self.hi_delta, axis=0) * learning_rate
        # 更新输出层的偏置
        self.output_bias += np.sum(self.ou_delta, axis=0) * learning_rate        
