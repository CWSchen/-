'''
Python3与OpenCV3.3 图像处理(十)--EPF
https://blog.csdn.net/gangzhucoll/article/details/78700989

2017年12月03日 13:56:19
阅读数：390
一、什么是EPF
高斯模糊只考虑了权重，只考虑了像素空间的分布，没有考虑像素值和另一个像素值之间差异的问题，如果像素间差异较大的情况下（比如图像的边缘），高斯模糊会进行处理，但是我们不需要处理边缘，要进行的操作就叫做边缘保留滤波（EPF）
二、示例
'''
import cv2 as cv
import numpy as np

def bi(image):
    """
    色彩窗的半径
    图像将呈现类似于磨皮的效果
    """

    #image：输入图像，可以是Mat类型，
    #       图像必须是8位或浮点型单通道、三通道的图像
    #0：表示在过滤过程中每个像素邻域的直径范围，一般为0
    #后面两个数字：空间高斯函数标准差，灰度值相似性标准差
    dst=cv.bilateralFilter(image,0,60,10);
    cv.imshow('bi',dst)

def shift(image):
    """
    均值迁移
    图像会呈现油画效果
    """

    #10:空间窗的半径
    #50:色彩窗的半径
    dst=cv.pyrMeanShiftFiltering(image,10,50);
    cv.imshow('shift',dst)



pic_1 = '0-common_pics/common_1.png'
src=cv.imread(pic_1)

#图一（原图）
cv.imshow('def',src)
#图二（色彩窗的半径）
bi(src)
#图三（均值迁移）
shift(src)
cv.waitKey(0)

cv.destroyAllWindows()