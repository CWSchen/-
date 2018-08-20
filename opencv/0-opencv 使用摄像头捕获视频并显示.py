'''
https://blog.csdn.net/huanglu_thu13/article/details/52337013
'''
'''
概述

本节实现的是使用内建摄像头捕获视频，并显示视频的每一帧以实现视频的播放。

创建摄像头对象
逐帧显示实现视频播放
实现过程

引用

不再赘述，代码如下。

import cv2
import numpy
import matplotlib.pyplot as plot
1
2
3
创建摄像头对象

使用opencv自带的VideoCapture()函数定义摄像头对象，其参数0表示第一个摄像头，一般就是笔记本的内建摄像头。

cap = cv2.VideoCapture(0)
1
逐帧显示实现视频播放

在while循环中，利用摄像头对象的read()函数读取视频的某帧，并显示，然后等待1个单位时间，如果期间检测到了键盘输入q，则退出，即关闭窗口。

while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
1
2
3
4
5
6
7
释放摄像头对象和窗口

调用release()释放摄像头，调用destroyAllWindows()关闭所有图像窗口。

cap.release()
cv2.destroyAllWindows()
1
2
源代码

整个程序的源代码如下：

# created by Huang Lu
# 27/08/2016 17:05:45
# Department of EE, Tsinghua Univ.

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

运行结果

在命令行进入该源程序所在目录后，运行python main.py后即可显示结果。显示结果如下：

![使用摄像头捕获视频并显示](https://raw.githubusercontent.com/hlthu/Python/master/Python_OpenCV/Capture_Video/Screenshot .png)

参考

OpenCV-Python-Toturial-中文版.pdf
https://github.com/hlthu/Python/tree/master/Python_OpenCV/Capture_Video/
https://hlthu.github.io/opencv/2016/06/09/python-opencv-7.html
'''