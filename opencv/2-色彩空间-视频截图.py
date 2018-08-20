import cv2 as cv
import numpy as np
'''
Python3与OpenCV3.3 图像处理（四）--色彩空间
https://blog.csdn.net/gangzhucoll/article/details/78574856

2017年11月19日 17:25:03
阅读数：733
一、本节简述

本节讲解图像色彩空间的处理和色彩空间的基础知识
二、色彩空间基础知识

什么是色彩空间，人们建立了多种色彩模型，以一维、二维、三维甚至四维空间坐标来表示某一色彩，
这种坐标系统所能定义的色彩范围即色彩空间

色彩空间有很多，但是常用的色彩空间一共5种：RGB、HSV、HSI、YCrCb、YUV，简单讲一下这5个色彩空间。

RGB就不用多说了，RGB是我门经常用到的；
HSV也称六角锥体模型，是根据颜色的直观特性创建的一种颜色空间，这个颜色空间是本节课讲解的一个重点。
HSI是从人的视觉系统出发，用色调（  Hue  ）、色饱和  度（  Saturation  或  Chroma  ）
和亮度（  Intensity  或  Brightness  ）来描述颜色。  HSI  颜色空间可以用一个圆  锥空间模型来描述
YCrCb主要用于优化彩色视频信号的传输，使其向后相容老式黑白电视，这个可以用来检测皮肤和检测人脸
YUV是被欧洲电视系统所采用的一种颜色编码方法（属于PAL），是PAL和SECAM模拟彩色电视制式采用的颜色空间。

三、色彩空间的转换

OpenCV提供多种将图像的色彩空间转换为另一个色彩空间的方法，转换方法的方法名一般为 “原色彩空间2需要转化的色彩空间”，
下面我们以图像的RGB色彩转换为其他四种色彩空间和GRAY色彩空间。
'''
def ColorSpace(image):
    """
    色彩空间转化
    RGB转换为其他色彩空间
    """
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    cv.imshow("gray",gray)
    hsv=cv.cvtColor(image,cv.COLOR_RGB2HSV)
    cv.imshow("hsv",hsv)
    yuv=cv.cvtColor(image,cv.COLOR_RGB2YUV)
    cv.imshow("yuv",yuv)
    ycrcb=cv.cvtColor(image,cv.COLOR_RGB2YCrCb)
    cv.imshow("ycrcb",ycrcb)
    cv.waitKey(0)

imagePath = '0-common_pics/common_1.jpg'
src = cv.imread(imagePath)
ColorSpace(src)

'''
四、标记图像中的特定颜色

一般对颜色空间的图像进行有效处理都是在HSV空间进行的，然后对于基本色中对应的HSV分量需要给定一个严格的范围，下面是网友通过实验计算的模糊范围（准确的范围在网上都没有给出）。

H:  0 — 180

S:  0 — 255

V:  0 — 255

以下是不同颜色的HSV最大最小的范围：

以下代码是标注出图像中的黑色部分，黑色部分将以白色显示，其他颜色部分将以黑色显示，
颜色标注OpenCV 提供了一个方法，inRange()。该方法提供三个参数，
第一个参数是图像色彩空间即hsv值，第二个参数是hsv的最小查找范围，第三个参数是hsv的最大查找范围。
代码运行后，将会标注出图像的黑色部分。
'''
capture=cv.VideoCapture("0-common_pics/test_mp4.mp4")
while(True):
    ret,frame=capture.read()
    if ret==False:
        break
    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    lower_hsv=np.array([0,0,0])
    upperb_hsv = np.array([180, 255, 46])
    mask=cv.inRange(hsv,lowerb=lower_hsv,upperb=upperb_hsv)
    cv.imshow("video_mask", mask)
    cv.imshow("video",frame)
    c=cv.waitKey(40)
    if c==27:
        break

'''
    #打开0号摄像头，捕捉该摄像头实时信息
    #参数0代表摄像头的编号
    #有多个摄像头的情况下，可用编号打开摄像头
    #若是加载视频，则将参数改为视频路径，cv.VideoCapture加载视频是没有声音的，OpenCV只对视频的每一帧进行分析
   capture=cv.VideoCapture(0)
'''