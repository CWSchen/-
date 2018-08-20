import cv2 as cv
import numpy as np
'''
Python3与OpenCV3.3 图像处理（五）--图像运算
https://blog.csdn.net/gangzhucoll/article/details/78598238

2017年11月21日 23:49:43
阅读数：570
一、本节简介

图像运算也就是像素运算，简单的说就是利用算术运算或逻辑运算，对图像的每个像素进行处理（例如两个图像的合并）。虽然我们可以像第二节课那样，一个像素一个像素的遍历并修改值，但是如果图像分辨率很大的情况下，会处理的很慢，并且处理一些复杂的运算时，我们的代码效率会变得更低，代码编写出来也变得很麻烦。这节课就来讲解以下OpenCV中对图像运算的方法。

注意：我们在处理两个图像时，图像的像素大小和类型要完全一致，否则OpenCV就会报错。

二、算术运算

图像算术运算就是对两个图像的每个像素点执行加减乘除的运算，从而得到一个新的图像。代码如下
'''
def add(image1,image2):
    """图片相加"""
    dst=cv.add(image1,image2)
    cv.imshow("add image",dst)


def subtract(image1,image2):
    """图片相减"""
    dst=cv.subtract(image1,image2)
    cv.imshow("subtract image",dst)


def divide(image1,image2):
    """图片相除"""
    dst=cv.divide(image1,image2)
    cv.imshow("divide image",dst)


def multiply(image1,image2):
    """图片相乘"""
    dst=cv.multiply(image1,image2)
    cv.imshow("multiply image",dst)
'''
三、逻辑运算

图像的逻辑运算就是对图像的每个像素点执行与或非的运算，从而得到一个新的图片，代码如下
'''
def logic(image1,image2):
    """逻辑运算"""
    #与操作
    dst=cv.bitwise_and(image1,image2)
    cv.imshow("logic",dst)
    # 或操作(与相加操作类似)
    dst = cv.bitwise_or(image1, image2)
    cv.imshow("logic", dst)
    # 非操作(像素取反)
    dst = cv.bitwise_not(image1)
    cv.imshow("logic", dst)

'''
四、其他算数运算
'''
def others(image1,image2):
    #计算每个通道的平均值
    m1= cv.mean(image1)
    m2 = cv.mean(image2)
    #计算每个通道的平均值和方差
    m1,dev1=cv.meanStdDev(image1)
    m2,dev2=cv.meanStdDev(image2)
    print(m1,dev1)
    print(m2,dev2)

'''
五、简单的Demo
'''
def contrast_brightness(image,c,b):
    """
    修改亮度和对比度
    c：对比度
    b：亮度
    """
   #获取图片的高、宽和通道数
    h,w,ch=image.shape
    #创建一个全黑色的图片
    blank=np.zeros([h,w,ch],image.dtype)
    #调整亮度和对比度
    dst=cv.addWeighted(image,c,blank,1-c,b)
    cv.imshow("con-bri",dst)

