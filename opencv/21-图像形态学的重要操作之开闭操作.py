import cv2 as cv
import numpy as np

'''
https://blog.csdn.net/gangzhucoll/article/details/78927295

一、什么是开操作和闭操作

闭操作：

 1、图像形态学的重要操作之一，基于膨胀与腐蚀操作组合形成的

 2、主要是应用在二值图像分析中，灰度图像也可以

 3、开操作=膨胀+腐蚀，输入图像+结构元素

开操作：

 1、图像形态学的重要操作之一，基于膨胀与腐蚀操作组合形成的

 2、主要是应用在二值图像分析中，灰度图像也可以

 3、开操作=腐蚀+膨胀，输入图像+结构元素

开操作与闭操作的区别是：膨胀与腐蚀的顺序

开操作作用：消除图像中小的干扰区域

闭操作作用：填充小的封闭区域
'''

def open(img):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    kernel=cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    #形态学操作
    #第二个参数：要执行的形态学操作类型，这里是开操作
    binary=cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    cv.imshow("open",binary)

def close(img):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    kernel=cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    #形态学操作
    #第二个参数：要执行的形态学操作类型，这里是开操作
    binary=cv.morphologyEx(binary,cv.MORPH_CLOSE,kernel)
    cv.imshow("close",binary)

common_path = '0-common_pics/num.jpg'
src=cv.imread(common_path)
cv.imshow('def',src)
open(src)
close(src)
cv.waitKey(0)
cv.destroyAllWindows()


'''
开闭操作（补充）
https://blog.csdn.net/gangzhucoll/article/details/78956799
一、顶帽
原图像与开操作之间的差值图像

二、黑帽
闭操作图像与原图像的差值图像

三、形态学梯度
1、基本梯度：膨胀后的图像减去腐蚀后的图像得到的差值图像
2、内部梯度：原图像减去腐蚀之后的图像得到的差值图像
3、外部梯度：图像膨胀之后减去原图像得到的差值图像
'''

def hat(img):
    """顶帽/黑帽梯度"""
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    cv.imshow("topHat", dst)

    dst = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
    cv.imshow("blackHat", dst)


def base(img):
    """基本梯度"""
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(gray, cv.MORPH_GRADIENT, kernel)
    cv.imshow("base", dst)


def i_e(img):
    """内/外梯度"""
    kerenl = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dm = cv.dilate(img, kerenl)
    em = cv.erode(img, kerenl)
    # 内梯度
    dst1 = cv.subtract(img, em)
    # 外梯度
    dst2 = cv.subtract(dm,img)
    cv.imshow("intrenal", dst1)
    cv.imshow("external", dst2)


src=cv.imread(common_path)
cv.imshow('def', src)
hat(src)
base(src)
i_e(src)
cv.waitKey(0)
cv.destroyAllWindows()