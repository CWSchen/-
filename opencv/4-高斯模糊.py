'''
Python3与OpenCV3.3 图像处理(九）--高斯模糊
https://blog.csdn.net/gangzhucoll/article/details/78682492

2017年12月01日 00:13:04
阅读数：618
一、什么是高斯模糊

把要模糊的像素色值统计，用数学上加权平均的计算方法（高斯函数）得到色值，对范围、半径等进行模糊


二、高斯模糊的应用场景

一些美颜软件、美颜相机上的磨皮和毛玻璃特效基本上都是用的高斯模糊，并且大部分图像处理软件中都有高斯模糊的操作，除此之外，高斯模糊还具有减少图像层次和深度的功能


三、示例
'''
import cv2 as cv
import numpy as np


def clamp(pv):
    """防止颜色值超出颜色取值范围（0-255）"""
    if pv>255:
        return 255
    if pv<0:
        return 0
    else:
        return pv

def gaussian_noise(image):
    """高斯噪声"""
    h,w,c=image.shape

    for row in range(h):
        for col in range(w):
            #获取三个高斯随机数
            #第一个参数：概率分布的均值，对应着整个分布的中心
            #第二个参数：概率分布的标准差，对应于分布的宽度
            #第三个参数：生成高斯随机数数量
            s=np.random.normal(0,20,3)
            #获取每个像素点的bgr值
            b=image[row,col,0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            #给每个像素值设置新的bgr值
            image[row,col,0]=clamp(b+s[0])
            image[row, col, 0] = clamp(g + s[1])
            image[row, col, 0] = clamp(r + s[2])

    cv.imshow("noise",image)

imagePath = '0-common_pics/common_1.jpg'
src = cv.imread(imagePath)

gaussian_noise(src)
#给图片创建毛玻璃特效
#第二个参数：高斯核的宽和高（建议是奇数）
#第三个参数：x和y轴的标准差
dst=cv.GaussianBlur(src,(5,5),15)
cv.imshow("gaussian",dst)

#等待用户操作
cv.waitKey(0)
#释放所有窗口
cv.destroyAllWindows()