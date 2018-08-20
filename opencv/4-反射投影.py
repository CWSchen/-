'''
Python3与OpenCV3.3 图像处理(十三)--反射投影
2017年12月07日 00:44:58
阅读数：448
https://blog.csdn.net/gangzhucoll/article/details/78736768

一、什么是反射投影

简单的说就是通过给定的直方图信息，在图像找到相应的像素分布区域

二、反射投影的应用

物体跟踪、定位物体等
'''
import cv2 as cv
import numpy as np
from matplotlib import  pyplot as plt


Reflect_sample = "0-common_pics/Reflect_sample.jpg"
Reflect_sample_red = "0-common_pics/Reflect_sample_red.jpg"
Reflect_target = "0-common_pics/Reflect_target.jpg"

def hist2d(image):
    """2d 直方图计算和现实"""
    #转换为hsv色彩空间
    hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)
    #[180,256] bins 越多对每个像素细分的越厉害，会导致反响直方图的碎片化
    #[0,180,0,256]:hsv色彩空间中 h和s的取值范围，只是固定的
    hist=cv.calcHist([image],[0,1],None,[180,256],[0,180,0,256])
    #interpolation:差值方式
    plt.imshow(hist,interpolation='nearest')
    #直方图名字
    plt.title("2D hist")
    #图三
    plt.show()

def backProjection():
    """直方图反响投影"""
    #样本图片
    # sample=cv.imread(Reflect_sample)  # 不全是红色的样本图片
    sample=cv.imread(Reflect_sample_red) # 全是红色的样本图片
    #目标片图片
    target=cv.imread(Reflect_target)
    sample_hsv=cv.cvtColor(sample,cv.COLOR_BGR2HSV)
    target_hsv=cv.cvtColor(target,cv.COLOR_BGR2HSV)

    #图一
    cv.imshow("sample",sample)
    #图二
    cv.imshow("target",target)

    #获得样本图片直方图
    #[0,1]:用于计算直方图的通道，这里使用hsv计算直方图，所以就直接使用第一h和第二通道，即h和s通道；
    #None:是否使用mask，None 否
    #[32,32] bins 越多对每个像素细分的越厉害，会导致反响直方图的碎片化
    #[0,180,0,256]:hsv色彩空间中 h和s的取值范围，是固定的
    sample_hist=cv.calcHist([sample_hsv],[0,1],None,[32,32],[0,180,0,256])

    #规划样本图片直方图
    #sample_hist:输入的矩阵
    #sample_hist：归一化后的矩阵
    #0:归一化后的矩阵的最小值
    #255：归一化后的矩阵的最大值
    #cv.NORM_MINMAX:数组的数值被平移或缩放到一个指定的范围，线性归一化，一般较常用
    cv.normalize(sample_hist,sample_hist,0,255,cv.NORM_MINMAX)
    #生成反响投影
    #target_hsv:目标图像hsv矩阵
    #[0,1]:用于计算直方图反射投影的通道，这里使用hsv计算直方图，所以就直接使用第一h和第二通道，即h和s通道；
    # [0,180,0,256]:hsv色彩空间中 h和s的取值范围，是固定的
    #1:是否缩放大小，1不需要，0需要
    dst=cv.calcBackProject([target_hsv],[0,1],sample_hist,[0,180,0,256],1)
    #图四
    cv.imshow("bp",dst)



src=cv.imread(Reflect_target)

hist2d(src)
backProjection()

#等待用户操作
cv.waitKey(0)
#释放所有窗口
cv.destroyAllWindows()
