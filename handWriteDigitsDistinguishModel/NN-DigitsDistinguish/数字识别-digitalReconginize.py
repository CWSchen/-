# -*- coding:utf-8 -*-
# from sklearn.datasets import load_digits

# digits= load_digits()
# print(digits.data.shape)
#
# import pylab as pl
# pl.gray()
# pl.matshow(digits.images[0])
# pl.show()

# 每个图片8x8  识别数字：0,1,2,3,4,5,6,7,8,9 手写识别
from sklearn.preprocessing import LabelBinarizer
from NeuralNetwork import NeuralNetwork
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

digits = load_digits()
x = digits.data
y = digits.target
# 得到一个 交叉验证的比例
x -= x.min()
x /= x.max()

nn = NeuralNetwork([64, 100, 10], "logistic")
x_train, x_test, y_train, y_test = train_test_split(x, y)
print(x_test.shape)

'''
对于标称型数据，preprocessing.LabelBinarizer 标签二值化，是一个很好用的工具。
可把yes和no转化为0和1，或把 incident(变化) 和 normal (正常)  转化为0和1。
当然，对于两类以上的标签也是适用的。
'''
label_train = LabelBinarizer().fit_transform(y_train)
label_test = LabelBinarizer().fit_transform(y_test)
print("start fitting..")
predictions = []
nn.fit(x_train, label_train, epochs=10000)

# 官方的测试集
for i in range(x_test.shape[0]):
    o = nn.predict(x_test[i])
    predictions.append(np.argmin(o))

confusion_matrix(y_test, predictions)
classification_report(y_test, predictions)

print('预测的结果集\n',predictions)

# argmax 返回最大值第一次出现的 下标位置
print(predictions[np.argmax(predictions)])
print('最大值%d，是第%d个元素' % ( max(predictions), np.argmax(predictions) + 1 ))




