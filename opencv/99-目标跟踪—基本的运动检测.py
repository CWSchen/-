# coding=utf-8
import cv2
import numpy as np
'''
https://blog.csdn.net/qq_16463925/article/details/73733295

目标跟踪是对摄像头视频中的移动目标进行定位的过程，
一种简单的方法就是计算帧与帧之间的差异。本文利用“背景”zhen与其他帧之间的差异，来跟踪视频中的目标。
'''

#捕获视频图像
camera = cv2.VideoCapture(0)
#打开摄像头，将第一帧作为整个输入背景
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
kernel = np.ones((5, 5), np.uint8)
background = None

while (True):
    ret, frame = camera.read()
    #对背景帧进行灰度和平滑处理
    if background is None:
        background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background = cv2.GaussianBlur(background, (21, 21), 0)
        continue
        #将其他帧进行灰度处理和模糊平滑处理
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    # 计算其他帧与背景之间的差异，得到一个差分图
    diff = cv2.absdiff(background, gray_frame)
    # 应用阈值得到一副黑白图像，并通过dilate膨胀图像，从而对孔和缺陷进行归一处理
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    diff = cv2.dilate(diff, es, iterations=2)

    '''显示矩形框，在计算出的差分图中找到所有的白色斑点轮廓，并显示轮廓；计算一幅图像中目标的轮廓'''
    image, cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        if cv2.contourArea(c) < 1500:
            continue
        (x, y, w, h) = cv2.boundingRect(c) # 计算矩形的边框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv2.imshow("contours", frame)
    cv2.imshow("dif", diff)
    if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
        break

cv2.destroyAllWindows()
camera.release()

'''
说明：进行模糊处理是为了避免震动，光照等产生的噪声影响。进行平滑处理是避免运动和跟踪过程将其检测出来。
'''