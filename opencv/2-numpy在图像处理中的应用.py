'''
Python3与OpenCV3.3 图像处理（三）--Numpy数组操作
https://blog.csdn.net/gangzhucoll/article/details/78546290

2017年11月15日 23:25:38
阅读数：1338
一、本节简述

本节主要讲解Numpy数组操作的一些基础知识。


二、什么是Numpy

       一个用python实现的科学计算包。包括：1、一个强大的N维数组对象Array；2、比较成熟的（广播）函数库；3、用于整合C/C++和Fortran代码的工具包；4、实用的线性代数、傅里叶变换和随机数生成函数。numpy和稀疏矩阵运算包scipy配合使用更加方便。
NumPy（Numeric Python）提供了许多高级的数值编程工具，如：矩阵数据类型、矢量处理，以及精密的运算库。专为进行严格的数字处理而产生。多为很多大型金融公司使用，以及核心的科学计算组织如：Lawrence Livermore，NASA用其处理一些本来使用C++，Fortran等所做的任务。


三、示例代码
'''
import cv2 as cv
import  numpy as np


def access_pixel(image):
    """访问图像所有的像素"""
    print(image.shape)

    #获取图像的高度，图像的高度为shape的第一个值（维度）
    height=image.shape[0]
    #获取图像的宽读，图像的宽度为shape的第二个值（维度）
    width=image.shape[1]
    #获取图像通道数目，图像的通道数目为shape的第三个值（维度）
    #加载进来的图像都有三个通道，三个通道是图像的RGB
    channels=image.shape[2]
    print("width: %s,height: %s channels: %s"%(width,height,channels))

    ''' 操作像素 方法一：循环获取每个像素点，并且修改，然后存储修改后的像素点 '''
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv=image[row,col,c]
                image[row,col,c]=255-pv

    #输出的是一个呈现负片效果的图片
    cv.imshow("pixels_demo",image)


def access_pixel2(image):
    """
    访问图像所有的像素
    https://blog.csdn.net/gangzhucoll/article/details/78569619
    """

    '''操作像素 方法二：将图像呈现出负片的效果，也就是像素取反。
    这段代码本身是没有问题的，但是运行起来后会发现你所提供的图片越大，处理起来速度越慢，
    这是因为我们在代码里使用了嵌套多层for循环，对于我们练习这样是没问题的，
    但是如果在实际项目中这样使用，会导致系统运行特别慢。那么我们该怎么办呢，
    其实OpenCV中有像素取反的方法，只需把for循环代码改为如下一行代码，就可以了。
    这样即能实现负片效果，也缩短了处理图片像素的时间
    '''
    cv.bitwise_not(image)

    #输出的是一个呈现负片效果的图片
    cv.imshow("pixels_demo",image)


def create_image():
    """创建新图象"""
    #创建一张宽高都是400像素的3通道 8位图片
    img=np.zeros([400,400,3],np.uint8)
    #修改通道值
    img[:,:,0]=np.ones([400,400])*255
    img[:, :, 2] = np.ones([400, 400]) * 255
    cv.imshow("new image",img)

    #创建一个单通道的8位图片
    img=np.zeros([400,400,1],np.uint8)
    img=img*127
    cv.imshow("new image", img)
    cv.imwrite("127img.png",img)

    #numpy 数组维度的变换
    #定义一个二维数组
    img=np.ones([3,3],np.uint8)
    #填充每个元素
    img.fill(1000.22)
    print(img)
    #变换为一维数组
    img=img.reshape([1,9])
    print(img)

imagePath = '0-common_pics/common_1.jpg'
src = cv.imread(imagePath)

#获取cpu当前时钟总数
t1=cv.getTickCount()
access_pixel(src)
t2=cv.getTickCount()
#计算处理像素花费的时间
#cv.getTickFrequency() 每秒的时钟总数
time=((t2-t1)/cv.getTickFrequency())
print("time: %s s"%time)
create_image()
#等待用户操作
cv.waitKey(0)
#释放所有窗口
cv.destroyAllWindows()

