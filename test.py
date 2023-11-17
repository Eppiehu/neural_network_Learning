import numpy as np
from neural_network import NeuralNetwork

# 创建一个神经网络实例：输入层有3个神经元，隐藏层有3个神经元，输出层有1个神经元
nn = NeuralNetwork(3, 3, 1)

# 定义训练数据：输入和期望输出
# 这里的数据是一个简单的示例，您可以根据实际需要调整
training_in = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 0]])
training_ou = np.array([[0], [0], [1], [0], [1], [1], [0], [1]])

# 设置学习率和迭代次数（epochs）
learning_rate = 0.1
epochs = 20000

# 训练神经网络
for epoch in range(epochs):
    for inputs, ex_out in zip(training_in, training_ou):
        nn.forward(inputs)  # 进行前向传播
        nn.backward(inputs, ex_out, learning_rate)  # 进行反向传播和权重更新

# 测试神经网络
for inputs in training_in:
    prediction = nn.forward(inputs)  # 对每个输入进行预测
    print(f"Input: {inputs} Prediction: {prediction}")

                   
