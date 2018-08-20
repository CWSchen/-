'''
Python3与OpenCV3.3 图像处理(六）--ROI
https://blog.csdn.net/gangzhucoll/article/details/78609771

2017年11月23日 00:30:45
阅读数：1889
一、本节简介

本节主要讲解ROI的图像中特定区域的提取和合并图片


二、什么是ROI

简单的说就是对图像感兴趣的区域，机器视觉、图像处理中，从被处理的图像以方框、圆、椭圆、不规则多边形等方式勾勒出需要处理的区域，称为感兴趣区域，ROI。举个例子来说：有一副图片，图片上有各种动物i，但是你只喜欢图片里的狗，那么这个狗所在的区域就是感兴趣的区域（ROI）。

三、示例
'''
import cv2 as cv

imagePath = '0-common_pics/common_1.jpg'
src = cv.imread(imagePath)

cv.namedWindow('input image',cv.WINDOW_AUTOSIZE)
cv.imshow('input image',src)

#高度从42像素开始到282像素
#宽度从184像素开始到355像素
#高度起始位置是从图片的顶部算起，宽度起始位置是从图片的左侧算起
#本例中的起始位置和结束位置是通过PhotoShop 测量出来的，在实际应用中这两个位置是通过算法计算出来的
face=src[42:282,184:355]
#效果见图1，我们取出了原图的人脸
cv.imshow("取出的图像",face)

#将取出的区域改变为灰度图像
gray=cv.cvtColor(face,cv.COLOR_BGR2GRAY)

#将灰度图像变为RGB图像
#这里改变色彩空间的原因是灰度图像是单通道的，原图是三通道的，无法合并
#所以需要先转换为三通道的RGB色彩空间
backface=cv.cvtColor(gray,cv.COLOR_GRAY2BGR)

#将取出并处理完的图像和原图合并起来
src[42:282,184:355]=backface
#效果见图2
cv.imshow("合并后的图像",src)


cv.waitKey(0)
cv.destroyAllWindows()