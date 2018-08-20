# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
import cv2

'''''
19章 Canny边缘检测
步骤:
1噪声去除
使用5*5的高斯滤波器
2计算图像梯度
对平滑后的图像使用Sobel算子计算水平方向和竖直方向的一阶导数（图像梯度）
(Gx和Gy)。根据得到的这两幅梯度图(Gx和Gy)找到边界的梯度和方向，公式如下
Edge_Gradient(G)=sqrt( Gx*Gx + Gy*Gy )

Angle(θ)= tan −1( Gx/Gy)
梯度方向一般总是与边界垂直（？）,梯度方向被归为四类：垂直，水平，和两个对角线

3非极大值抑制
获得梯度方向和大小后，对整幅图像做扫描，去除非边界上的点。
对每一个像素检查，看这个点的梯度是不是周围具有相同梯度方向的
点中最大的

4滞后阈值
现在要确定那些边界才是真正的边界。这时我们需要设置两个阈值：
minVal 和 maxVal。当图像的灰度梯度高于 maxVal 时被认为是真的边界，
那些低于 minVal 的边界会被抛弃。如果介于两者之间的话，就要看这个点是
否与某个被确定为真正的边界点相连，如果是就认为它也是边界点，如果不是
就抛弃
'''

def Canny_test(img):
    '''''
    cv2.Canny(image,threshold1 , threshold2[,edges])
    作用：根据Canny算法检测边界
    threshold1:minVal，threshold2：maxVal
    edges:设置卷积核的大小,L2gradient:默认使用近似值来代替所求的梯度
    '''
    edges = cv2.Canny(img , 100 , 200)
    plt.subplot(1,2,1) , plt.imshow(img , cmap="gray")
    plt.title("Original") , plt.xticks([]) , plt.yticks([])
    plt.subplot(1,2,2), plt.imshow(edges , cmap="gray")
    plt.title("Cannt") , plt.xticks([])  ,plt.yticks([])
    plt.show()

if __name__ == "__main__":

    common_pics_path = "0-common_pics/common_1.jpg"
    img = cv2.imread(common_pics_path, 0) # 灰度图计算
    Canny_test(img)


'''
Python3与OpenCV3.3 图像处理(十八）--Canny边缘提取
2017年12月17日 15:30:21
阅读数：1256
一、什么是边缘检测

图像的边缘检测的原理是检测出图像中所有灰度值变化较大的点，而且这些点连接起来就构成了若干线条，这些线条就可以称为图像的边缘。


二、canny 算法五步骤

高斯模糊
灰度转换
计算梯度
非最大信号抑制
高低阈值输出二值图像
三、示例代码
'''
def edge(img):
    #高斯模糊,降低噪声
    blurred = cv2.GaussianBlur(img,(3,3),0)
    #灰度图像
    gray=cv2.cvtColor(blurred,cv2.COLOR_RGB2GRAY)
    #图像梯度
    xgrad=cv2.Sobel(gray,cv2.CV_16SC1,1,0)
    ygrad=cv2.Sobel(gray,cv2.CV_16SC1,0,1)
    #计算边缘
    #50和150参数必须符合1：3或者1：2
    edge_output=cv2.Canny(xgrad,ygrad,50,150)
    #图一
    cv2.imshow("edge",edge_output)

    dst=cv2.bitwise_and(img,img,mask=edge_output)
    #图二（彩色）
    cv2.imshow('cedge',dst)

common_pics_path = "0-common_pics/common_1.jpg"
src = cv2.imread(common_pics_path)
#图三（原图）
cv2.imshow('def',src)
edge(src)
cv2.waitKey(0)
cv2.destroyAllWindows()