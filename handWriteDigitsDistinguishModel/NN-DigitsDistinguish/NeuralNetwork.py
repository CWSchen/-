


# 1. 关于非线性转化方程(non-linear transformation function)
#
# sigmoid函数(S 曲线)用来作为activation function:
#
# 1.1 双曲函数(tanh)
#
# 1.2  逻辑函数(logistic function)
#
# 2. 实现一个简单的神经网络算法


import numpy as np

'''
sigmoid 函数 有如下两种 方法
tanh 双曲函数  和   logistic 逻辑函数
'''
def tanh(x):  # 双曲函数
    return np.tanh(x)


def tanh_deriv(x):  # 双曲函数导数
    return 1.0 - np.tanh(x) * np.tanh(x)


def logistic(x):  # sigmoid 函数
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):  # sigmoid 函数导数
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        for i in range(1, len(layers) - 1):  # add weights
            '''
            分别要 定义 前向 权重  和 后向 权重
            '''
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)
            '''
            Python是完全面向对象的，因此所有的数据都是对象
                random.random()生成0和1之间的随机浮点数float，
                    它其实是一个隐藏的random.Random类的实例的random方法。
                random.Random() 生成random模块里得Random类的一个实例，
                    这个实例不会和其他Random实例共享状态，一般是在多线程的情况下使用。
            '''

    def fit(self, X, y, learning_rate=0.2, epochs=10000):# fit 拟合
        X = np.atleast_2d(X)  # 转二维数组
        temp = np.ones([X.shape[0], X.shape[1] + 1]) # # 按照行列分别定义一个全是 1 的矩阵
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])  # x.shape[0] is the number of the trainingset
            a = [X[i]]  # choose a sample randomly to train the model

            for l in range(len(self.weights)):  # going forward network, for each layer
                # Computer the node value for each layer (O_i) using activation function
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = y[i] - a[-1]  # Computer the error at the top layer
            # For output layer, Err calculation (delta is updated error)
            deltas = [error * self.activation_deriv(a[-1])]

            # Start backprobagation 后向算法
            for l in range(len(a) - 2, 0, -1):
                #Compute the updated error (i,e, deltas) for each node going from top layer to input layer
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))

            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i]) # 转 二维数组
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a


'''
# 1. 简单 非线性关系 nonlineartest 数据集测试(XOR):
#
# X:                  Y
# 0 0                 0
# 0 1                 1
# 1 0                 1
# 1 1                 0


nn = NeuralNetwork([2, 2, 1], 'tanh')
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print(np.atleast_2d(X))
y = np.array([0, 1, 1, 0])
nn.fit(X, y)
for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    print(i, nn.predict(i))
'''