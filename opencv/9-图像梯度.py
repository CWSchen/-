# coding: utf-8
import cv2
from matplotlib import pyplot as plt
import numpy as np

'''''
18 图像梯度
https://blog.csdn.net/qingyuanluofeng/article/details/51594506
梯度是求导，三种梯度滤波器，（高通滤波器）:Sobel,Scharr和Laplacian
SobelScharr:求一阶或二阶导数，Scharr是对Sobel(使用小的卷积核求解梯度角度时)的优化。
Laplacian:是求二阶导数
Sobel 算子是高斯平滑u微分操作的结合体，抗噪声能力很好，可以设定求导的方向(xorder或yorder)
还可以设定使用的卷积核的大小(ksize)。如果ksize=-1,会使用3*3的Scharr滤波器，它的效果要
比3*3的Sobel滤波器好，3*3的Scharr滤波器卷积核如下:
        -3  0 3
x方向   -10 0 10
        -3 0 3

        -3 -10 -3
y方向    0  0  0
         3  10 3

Laplacian算子可以使用二阶导数的形式定义，假设其
离散实现类似于二阶Sobel导数，Opencv在计算拉普拉斯算子时
直接调用Sobel算子，计算公式为

△src =∂ 2 src
∂x 2
+
∂ 2 src
∂y 2
△src = ∂2 src / ∂ x2 + ∂2 src / ∂ y2
拉普拉斯滤波器使用的卷积核

        0   1   0
kernel=1    -4  1
        0   1   0
这些算子应该是用于做边缘检测的(高频成分)

注意：使用cv2.CV_64F的原因是从白到黑的边界点导数为负数后，如果
使用的是np.int8则会变成0，把边界地市
'''
def imageGradient_test(img):

    '''''
    cv2.Laplacian(src,ddepth)
    ddepth:目标图像要求的深度
    '''
    laplacian = cv2.Laplacian(img , cv2.CV_64F)
    '''''
    cv2.Sobel(src,ddepth,dx,dy[,ksize])
    作用：计算Sobel算子
    ddpeth:输出图像的深度，比如CV_8U,CV_64F等
    dx:x的导数，dy:y方向的导数
    ksize:核的代销，必须是1,3,5或7
    '''
    sobelx = cv2.Sobel(img , cv2.CV_64F , 1 , 0 , ksize=5)
    sobely = cv2.Sobel(img , cv2.CV_64F , 0 , 1 , ksize=5)

    plt.subplot(2,2,1) , plt.imshow(img , cmap="gray")
    plt.title("Original") , plt.xticks([]) , plt.yticks([])
    plt.subplot(2,2,2), plt.imshow(laplacian , cmap="gray")
    plt.title("Laplacian") , plt.xticks([])  ,plt.yticks([])
    plt.subplot(2,2,3), plt.imshow(sobelx , cmap="gray")
    plt.title("Sobelx") , plt.xticks([])  ,plt.yticks([])
    plt.subplot(2,2,4), plt.imshow(sobely , cmap="gray")
    plt.title("Sobely") , plt.xticks([])  ,plt.yticks([])
    plt.show()

if __name__ == "__main__":

    common_pics_path = "0-common_pics/common_1.jpg"
    img = cv2.imread(common_pics_path, 0) # 灰度图计算
    imageGradient_test(img)

'''
一、什么是图像梯度

可以把图像看成二维离散函数，图像梯度其实就是这个二维离散函数的求导：
图像梯度: G(x,y) = dx i + dy j;
dx(i,j) = I(i+1,j) - I(i,j);
dy(i,j) = I(i,j+1) - I(i,j);
其中，I是图像像素的值(如：RGB值)，(i,j)为像素的坐标。
图像梯度一般也可以用中值差分：
dx(i,j) = [I(i+1,j) - I(i-1,j)]/2;
dy(i,j) = [I(i,j+1) - I(i,j-1)]/2;
图像边缘一般都是通过对图像进行梯度运算来实现的。

图像梯度的最重要性质是，梯度的方向在图像灰度最大变化率上，它恰好可以反映出图像边缘上的灰度变化
上面说的是简单的梯度定义，其实还有更多更复杂的梯度公式。（来源百度）
'''
import cv2 as cv

def sobel(img):
    """索贝尔算子"""
    grad_x=cv.Sobel(img,cv.CV_32F,1,0)
    grad_y=cv.Sobel(img,cv.CV_32F,0,1)
    gradx=cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("x",gradx)
    cv.imshow("y",grady)

    gradxy=cv.addWeighted(gradx,0.5,grady,0.5,0)
    cv.imshow("grad",gradxy)

def scharr(img):
    """某些边缘差异很小的情况下使用"""
    grad_x = cv.Scharr(img, cv.CV_32F, 1, 0)
    grad_y = cv.Scharr(img, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("x", gradx)
    cv.imshow("y", grady)

    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow("grad", gradxy)

def lapalian(img):
    """拉普拉斯算子"""
    dst=cv.Laplacian(img,cv.CV_32F)
    lpls=cv.convertScaleAbs(dst)
    cv.imshow("lpls",lpls)


common_pics_path = "0-common_pics/common_1.jpg"
src = cv2.imread(common_pics_path)
lapalian(src)
cv.waitKey(0)
cv.destroyAllWindows()