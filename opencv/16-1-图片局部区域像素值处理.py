# 从Opencv中导入函数

import cv2 as cv
'''
来源：https://blog.csdn.net/lwplwf/article/details/57404974

背景故事：我需要对一张图片做一些处理，是在图像像素级别上的数值处理，
以此来反映图片中特定区域的图像特征，网上查了很多，大多关于opencv的应用教程帖子基本是停留在打开图片，
提取像素重新写入图片啊之类的基本操作，我是要取图片中的特定区域再提取它的像素值，
作为一个初学者开始接触opencv简直一脸懵逼，
慢慢摸索着知道了opencv的一些函数是可以实现的像SetImageROI()函数设置ROI区域，即感兴趣区域，就很好用啊，
总之最后是实现了自己想要的功能。现在看个程序确实是有点挫，也有好多多余的没必要的代码，
但毕竟算一次码代码的历程，就原模原样贴在这里吧。

代码功能：在python下用opencv，
- 打开图片并显示并重新写入新的文件
- 提取图片特定区域的像素值（根据自己需求，下面在代码中注解）
- 对提取出来的像素值做处理用matplotlib显示成条形图

https://blog.csdn.net/mikedadong/article/details/51264759
'''
# -*- coding:utf-8 -*-
__author__ = 'lwp'
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = "0-common_pics/common_1.jpg"
lwpImg = cv.imread(path)
# 创建图像空间，参数为size, depth, channels，这里设置的是图片等高宽30个像素的一个区域，8位，灰度图
# box_lwpImg = cv.create((30, 576), 8, 1)   # cv2 中没有 create方法了，用np.zeros 创建矩阵
box_lwpImg = np.zeros((30, 576), dtype=np.uint8 )

# 创建窗口
cv.namedWindow('test1', cv.WINDOW_AUTOSIZE)
cv.namedWindow("box_test1", cv.WINDOW_AUTOSIZE)

'''
# 设置ROI区域，即感兴趣区域，参数为x, y, width, heigh
cv.setImageROI(lwpImg, (390, 0, 30, 576))  # 老版本使用

cv2 利用numpy中的数组切片 设置ROI区域
'''
roiImg = lwpImg[390:0, 576:30]

'''
# 提取ROI，从lwpImg图片的感兴趣区域到box_lwpImg
cv.Copy(lwpImg, box_lwpImg)
cv.copy 方法已经不用了
'''

# 对box区域进行循环提取像素值存到列表pixel_list中
pixel_list = []
for i in range(30): # 576为box的高
    for j in range(576): # 30为box的宽
        x = box_lwpImg[i, j]
        pixel_list.append(x)

# 提取的像素值转为int整型赋给一维数组pixel_list_np_1
pixel_list_np_1 = np.array(pixel_list, dtype=int)
# 转为576*30的二位数组，即按图片box排列
pixel_list_np_2 = np.array(pixel_list_np_1).reshape(576, 30)
# 行求和，得到576个值，即每行的像素信息
pixel_sum = np.sum(pixel_list_np_2, axis=1)

# 取消设置
# cv.ResetImageROI(lwpImg)

# 画目标区域 cv.rectangle( img,左上点坐标,右下点坐标,颜色BGR值,线宽 )
lwpImg = cv.rectangle(lwpImg, (390, 0), (425, 576), (0, 0, 255), 2)
'''
OpenCV中绘制基本几何图形【矩形rectangle()、椭圆ellipse() 、圆circle() 】
'''
# 显示图像
cv.imshow('test1', lwpImg)
# 查看列表list长度，以确定像素值提取准确
list_length = len(pixel_list)
print( list_length)

# 查看数组维度，shape验证
print( pixel_list_np_1.ndim)
print( pixel_list_np_1.shape)
# print( pixel_list_np_1)

print( pixel_list_np_2.ndim)
print( pixel_list_np_2.shape)
# print( pixel_list_np_2)

# print( pixel_sum)

# 画条形图
plt.figure(1)
width = 1
for i in range(len(pixel_sum)):
    plt.figure(1)
    plt.bar(i, pixel_sum[i], width)
plt.xlabel("X")
plt.ylabel("pixel_sum")
plt.show()

# 按ESC退出，按s保存图片
k = cv.waitKey(100)
if k == 27:                 # wait for ESC key to exit
    cv.DestroyAllWindows()
elif k == ord('s'):         # wait for 's' key to save and exit
    cv.WriteFrame('copy_test.png', lwpImg)
    cv.DestroyAllWindows()

