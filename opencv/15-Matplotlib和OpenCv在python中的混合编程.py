import numpy as np # [1]导入python中的数值分析,矩阵运算的程序库模块
import cv2  # [2]导入OpenCv程序库模块
from matplotlib import pyplot as plt  # [3]仅仅导入了matplotlib绘图程序库中的一个子模块

'''
(一)使用Matplotlib
Matplotlib是python的一个绘图程序库,里面有各种各样的绘图方法,这个程序库相当于Matlab中的二维和三维绘图的
部分.在后面我们会继续深入的用到Matplolib这个绘图程序库。
在这节,我们学习一下,如何使用Matplotlib中的函数显示图像,如下所示:
'''
image_path = "0-common_pics/common_1.jpg"

srcImg = cv2.imread(image_path, 0)  # [1]以三通道彩色图片的形式,将图片加载到内存
plt.imshow(srcImg, cmap="gray", interpolation="bicubic")
plt.xticks([])  # [2]隐藏ｘ和ｙ上的值
plt.yticks([])
plt.show()
'''
注意的一点是：
    Opencv      是以彩色图片 BGR 模式加载图片。
    Matplotlib  是以彩色图片 RGB 模式加载的。
所以,彩色图片如果已经被OpenCv读取,那么,它将不会被正确显示.正因为这样,此处才加载和显示的是灰度图
'''