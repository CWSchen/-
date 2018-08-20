'''
Python3与OpenCV3.3 图像处理(十五)--图像二值化
https://blog.csdn.net/gangzhucoll/article/details/78760747
2017年12月09日 18:40:31
阅读数：256
一、什么是二值图像
图像中只有0和1，即1表示黑色，0表示白色

二、图像二值化的方法
图像二值化的方法：全局阈值，局部阈值。一般来说局部阈值要优于全局阈值。
在OpenCV中图像二值化的方法有OTS,Triangle,自动与手动，
衡量阈值方法是否是符合场景的，就是要看处理之后图像的信息是否丢失
'''
import cv2 as cv
import numpy as np

def threshold(image):
    """图像二值化：全局阈值"""
    #图像灰度化
    gray=cv.cvtColor(image,cv.COLOR_RGB2GRAY)
    #变为二值图像
    #gary：灰度图像
    #0：阈值，如果选定了阈值方法，则这里不起作用
    ret ,binary=cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    print(ret)
    cv.imshow("binary",binary)


def local_threshold(image):
    """局部阈值"""
    # 图像灰度化
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # 变为二值图像
    binary = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,25,10)
    cv.imshow("local_threshold", binary)

def custom_threshold(image):
    """局部阈值"""
    # 图像灰度化
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    h,w=gray.shape[:2]
    m=np.reshape(gray,[1,w*h])
    mean=m.sum()/(w*h)
    # 变为二值图像
    binary = cv.threshold(gray,mean,255,cv.THRESH_BINARY)
    cv.imshow("custom_threshold", binary)


common_pics_path = "0-common_pics/common_1.jpg"
src = cv.imread(common_pics_path)

threshold(src)

cv.waitKey(0)
cv.destroyAllWindows()

'''
补充
https://blog.csdn.net/gangzhucoll/article/details/78768283

在图片比较大的情况下，使用第十五节讲的方法，会出现处理速度慢和处理效果不佳的情况。
对于超大图象二值化一般都会进行分块。超大图象一般会分块以后使用全局二值化
，或者使用局部二值化。并且应使用自适应阈值，全局阈值会收到图象噪声的影响代码如下
'''


def big_img_binary(img):
    # 定义分割块的大小
    cw = 256
    ch = 256
    h,w = img.shape[:2]
    # 将图片转化为灰度图片
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    for row in range(0,h,ch):
        for col in range(0,w,cw):
            roi = gray[row:row+ch,col:col+cw]
            dst = cv.adaptiveThreshold(roi,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,127,20)
            gray[row:row+ch,col:col+cw]=dst
    # cv.imwrite('E:/rb.png',gray)
    cv.imwrite('E:/rb.png',gray)


common_pics_path = "0-common_pics/common_1.jpg"
src = cv.imread(common_pics_path)

big_img_binary(src)
cv.waitKey(0)
cv.destroyAllWindows()

