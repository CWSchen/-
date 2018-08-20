
import cv2 as cv

'''
https://blog.csdn.net/gangzhucoll/article/details/78868703

Python3与OpenCV3.3 图像处理(二十一）--轮廓发现
2017年12月21日 23:02:56
阅读数：781
一、什么是轮廓发现

是基于图像边缘提取的基础，寻找对象轮廓的方法，所以边缘提取的阈值选定会影响最终轮廓的发现

二、轮廓发现API

findContours 发现轮廓

drawContours绘制轮廓
'''

def contours(img):
    dst=cv.GaussianBlur(img,(3,3),0)
    #转换为灰度图像
    gray=cv.cvtColor(dst,cv.COLOR_RGB2GRAY)
    #转换为二值图像
    ret,binary=cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    cv.imshow("bi",binary)

    cloneImg,contours,heriachy= cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for i ,contour in enumerate(contours):
        cv.drawContours(img,contours,i,(0,0,255),2)
    cv.imshow("contpurs",img)


common_pics_path = "0-common_pics/common_1.jpg"
src = cv.imread(common_pics_path)
cv.imshow('def',src)
contours(src)
cv.waitKey(0)
cv.destroyAllWindows()


