'''
Python3与OpenCV3.3 图像处理(八）--模糊
https://blog.csdn.net/gangzhucoll/article/details/78660422

2017年11月28日 23:31:10
阅读数：692
一、模糊方式以及每种方式的使用场景

模糊操作方式：

均值模糊：一般用来处理图像的随机噪声
中值模糊：一般用来处理图像的椒盐噪声
自定义模糊：对图像进行锐化之类的操作

二、模糊基本原理

基于离散卷积、定义好每个卷积核、不同卷积核得到不同的卷积效果、模糊是卷积的一种表象


三、代码示例
'''
import cv2 as cv
import numpy as np


def blur(image):
    """
    均值模糊
    """
    #参数（5，5）：表示高斯矩阵的长与宽都是5
    dst=cv.blur(image,(5,5))
    #图二为均值模糊图
    cv.imshow("blur",dst)


def median(image):
    """
    中值模糊
    """
    #第二个参数是孔径的尺寸，一个大于1的奇数。
    # 比如这里是5，中值滤波器就会使用5×5的范围来计算。
    # 即对像素的中心值及其5×5邻域组成了一个数值集，对其进行处理计算，当前像素被其中值替换掉。
    #参考自：http://blog.csdn.net/sunny2038/article/details/9155893
    dst = cv.medianBlur(image, 5)
    #图三为中值模糊
    cv.imshow("median", dst)


def custom(image):
    """
    自定义模糊
    """
    #定义一个5*5的卷积核
    kernel=np.ones([5,5],np.float32)/25
    dst = cv.filter2D(image,-1,kernel=kernel)
    #图四为效果图
    cv.imshow("custom", dst)


imagePath = '0-common_pics/common_1.jpg'
src = cv.imread(imagePath)

#图一为原图
cv.imshow('image 1',src)

blur(src)
median(src)
custom(src)
#等待用户操作
cv.waitKey(0)
#释放所有窗口
cv.destroyAllWindows()