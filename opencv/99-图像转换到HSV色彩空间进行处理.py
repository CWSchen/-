import cv2
import numpy as np
'''
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces
'''

cap = cv2.VideoCapture(0)  # 视频捕捉

while(1):  # while true 不断捕捉视频

    # Take each frame
    # _, frame = cap.read()  # 读取视频捕捉

    imagePath = '0-common_pics/common_1.jpg'
    frame = cv2.imread(imagePath)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27: # 27 是 ESC
        break

cv2.destroyAllWindows()