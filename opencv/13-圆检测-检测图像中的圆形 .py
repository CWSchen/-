
import cv2 as cv

'''
https://blog.csdn.net/gangzhucoll/article/details/78858530
'''
import cv2 as cv
import numpy as np

def detect_circles(img):
    dst=cv.pyrMeanShiftFiltering(img,10,100)
    cimg=cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
    circles=cv.HoughCircles(cimg,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
    circles=np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv.circle(img,(i[0],i[1]),i[2],(0,0,255),2)
        cv.circle(img, (i[0], i[1]), 2, (255,0,0), 2)
    cv.imshow("c",img)

common_pics_path = "0-common_pics/common_1.jpg"
src = cv.imread(common_pics_path)
cv.imshow('def',src)
detect_circles(src)
cv.waitKey(0)
cv.destroyAllWindows()



