'''
Python3与OpenCV3.3 图像处理(十六）--图像金字塔
2017年12月12日 23:44:08
阅读数：407
https://blog.csdn.net/gangzhucoll/article/details/78787377

一、什么是图像金字塔
图像金字塔是图像多尺度表达的一种，是一种以多分辨率来解释图像的有效但概念简单的结构。一幅图像的金字塔是一系列以金字塔形状排列的分辨率逐步降低，且来源于同一张原始图的图像集合。其通过梯次向下采样获得，直到达到某个终止条件才停止采样。我们将一层一层的图像比喻成金字塔，层级越高，则图像越小，分辨率越低。（来源于百度）

二、图像金字塔类型
高斯金字塔
拉普拉斯金字塔
'''
import cv2 as cv
import numpy as np

def pyramin(img):
    """高斯金字塔"""
    #图像金字塔层数
    level=3
    #复制图片
    tmp=img.copy()
    pyramin_img=[]
    for i in range(level):
        dst=cv.pyrDown(tmp)
        pyramin_img.append(dst)
        cv.imshow("pyramid_down_"+str(i),dst)
        tmp=dst.copy()
    return pyramin_img

def lapalian(img):
    """拉普拉斯金字塔"""
    pyramid_images=pyramin(img)
    level=len(pyramid_images)
    print(level)
    #从高到低进行循环
    for i in range(level-1,-1,-1):
        if i == -1:
            #如果是第一幅图，则用原图进行计算
            exapand = cv.pyrUp(pyramid_images[i], dstsize=img.shape[:2])
            lpls = cv.subtract(img, exapand)
            cv.imshow("lpls_down_" + str(i), lpls)
        else:
            print(pyramid_images[i-1].shape[:2])
            help(cv.pyrUp)
            exapand=cv.pyrUp(pyramid_images[i],dstsize=pyramid_images[i-1].shape[:2])
            lpls=cv.subtract(pyramid_images[i-1],exapand)
            cv.imshow("lpls_down_"+str(i),lpls)


common_pics_path = "0-common_pics/common_1.jpg"
src = cv.imread(common_pics_path)
cv.imshow("def",src)
# pyramin(src)
lapalian(src)

cv.waitKey(0)
cv.destroyAllWindows()