import numpy as np  # [1]导入python中的数值分析,矩阵运算的程序库模块
import cv2  # [2]导入OpenCv程序库模块
from matplotlib import pyplot as plt  # [3]仅仅导入了matplotlib绘图程序库中的一个子模块

'''
# 1--我们可以根据像素的行和列的坐标获取它们的像素值
# 2--对BGR图像而言,返回的是B,G,R的像素值
# 3--对灰度图像而言,返回的是灰度值
'''
image_path = "0-common_pics/common_1.jpg"
srcImg = cv2.imread(image_path, 1)  #[1]彩色图片,其本质就是一个三维数组,所以可以用numpy中的方法操作

pixel = srcImg[100,100]  #[2]使用数组下标索引的方式,访问图片上(100,100)这一点像素的BGR值

print(" pixel value:", pixel) # pixel value: [60 20 32]

bule = srcImg[100, 100, 0]  #[3]返回这一点B的分量像素值
print("B pixel value:", bule) # B pixel value: 60

green = srcImg[100, 100, 1]  #[4]返回这一点G的分量像素值
print("G pixel value:", green) # G pixel value: 20

red = srcImg[100, 100, 2]  #[5]返回这一点R的分量像素值
print("R pixel value:", red) # R pixel value: 32



'''
注意:
    1--Numpy是经过优化了的进行快速矩阵运算的软件包
    2--所以,我们不推荐逐个获取像素值并修改,这样会很慢,能用矩阵运算,就不要使用循环
'''

'''
【模块２】获取图像的属性
    #1--图像的属性包括:
    #   1--行
    #   2--列
    #   3--通道
    #   4--图像的数据类型
    #   5--像素数目等等
    #2--srcImg.shape---可以获取图像的形状,返回一个图像的(行数,列数,通道数)Tuple
    #3--srcImg.size----可以返回图像的像素数目
    #4--srcImg.dtype---可以返回图像的数据类型
    #小结:
    #   其实,在numpy中,这些获取图片的属性方法,同样也是用于获取numpy中的数组中的相应属性
'''
print("Shape of the image--(row,column,channel):", srcImg.shape) #  (540, 1000, 3)
print("the number of the pixel of the image:", srcImg.size) # 1620000 = 540*1000*3
print("DataType of the image:", srcImg.dtype) #uint8   0 - 255
