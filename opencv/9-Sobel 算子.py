import numpy as np
import cv2  # opencv
'''
Python OpenCV -- Sobel 算子（九） -- 是一种带有方向性的滤波器，
http://blog.csdn.net/sunny2038/article/details/9170013

在Python 中的原型：
dst = cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]])

参数：
src -- 原图像
ddepth -- 图像的深度，-1 表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度。
dx dy -- 表示的是示导的阶数，0 表示这个方向上没有求导，一般为 0，1，2。

可选参数：
dst --  目标图像，与原图像（src）据有相同的尺寸和通道
ksize -- Sobel算子的大小，必须为1、3、5、7。
scale  --  缩放导数的比例常数，默认情况下没有伸缩系数
delta -- 一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中
borderType -- 判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。
'''
# help( cv2.Sobel )
img = cv2.imread("0-common_pics/common_1.jpg", 0)
x = cv2.Sobel(img,cv2.CV_16S,1,0) #
y = cv2.Sobel(img,cv2.CV_16S,0,1)

absX = cv2.convertScaleAbs(x)   # 转回uint8
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX,0.5,absY,0.5,0)

cv2.imshow("X", absX)
cv2.imshow("Y", absY)
cv2.imshow("Result", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()

'''
cv2.CV_16S -- Sobel 函数求完导数后会有负值和大于255的值，而原图像是uint8（8位无符号数据），所以在建立图像时长度不够，会被截断，所以使用16位有符号数据。
convertScaleAbs() --  转回uint8形式，否则将无法显示图像，而只是一副灰色图像
addWeighted() --  组合图像
函数原型：dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
src -- 原图像
dst -- 目标图像
alpha  -- 伸缩系数
beta  --  累加到结果上的一个值

dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])

alpha  --  第一幅图片中元素的权重
beta  --  第二个权重
gamma   --  累加到结果上的一个值
'''



'''
Python OpenCV -- Laplacian 算子（十）
https://blog.csdn.net/mokeding/article/details/19520833

Laplace算子的函数原型：
dst = cv2.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]])

src  --  原图像
ddepth -- 图像的深度， -1 表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度

【可选参数】

dst -- 目标图像

ksize --  算子的大小，必须为1、3、5、7。默认为1

scale  --  是缩放导数的比例常数，默认情况下没有伸缩系数

delta  --  是一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中

borderType  --  是判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。。
'''

gray_lap = cv2.Laplacian(img,cv2.CV_16S,ksize = 3)
dst = cv2.convertScaleAbs(gray_lap)

cv2.imshow('laplacian',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
