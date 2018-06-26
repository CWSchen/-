from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np
import scipy
import cv2
from fractions import Fraction

'''
sklearn内置了手写数字的数据集digits。此数据集的官方介绍在这里。摘录如下：

Each datapoint is a 8x8 image of a digit.

Classes	10
Samples per class	~180
Samples total	1797
Dimensionality	64
Features	integers 0-16
整体步骤分为：训练——预测两大步。用到的预测图片如下：
'''
def image2Digit(image):
    # 调整为8*8大小
    im_resized = scipy.misc.imresize(image, (8,8))
    # RGB（三维）转为灰度图（一维）
    im_gray = cv2.cvtColor(im_resized, cv2.COLOR_BGR2GRAY)
    # 调整为0-16之间（digits训练数据的特征规格）像素值——16/255
    im_hex = Fraction(16,255) * im_gray
    # 将图片数据反相（digits训练数据的特征规格——黑底白字）
    im_reverse = 16 - im_hex
    return im_reverse.astype(np.int)
# 加载数字数据
digits = datasets.load_digits()
# 划分训练集与验证集
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=2)
# 创建模型
clf = LogisticRegression(penalty='l2')
# 拟合数据训练
clf.fit(Xtrain, ytrain)
# 预测验证集
ypred = clf.predict(Xtest)
# 计算准确度
accuracy = accuracy_score(ytest, ypred)
print("识别准确度：",accuracy)

# 读取单张自定义手写数字的图片
image = scipy.misc.imread("examplePictures/1/7.jpg")
# 将图片转为digits训练数据的规格——即数据的表征方式要统一
im_reverse = image2Digit(image)
# 显示图片转换后的像素值
print(im_reverse)
# 8*8转为1*64（预测方法的参数要求）
reshaped = im_reverse.reshape(1,64)
# 预测
result = clf.predict(reshaped)
print(result)
'''
注意：

自定义图片最好是png格式，因为jpg采用的是有损压缩算法，图像数据会变化；
训练数据与预测数据格式需要一致，即特征一致；
上述代码基本上是对sklearn算法的简单调用，识别鲁棒性不高，所以图片中数字要很粗——方便识别
'''