import cv2 as cv
import numpy as np

'''
https://blog.csdn.net/u011321546/article/details/79645847

准备工作：找到分类器：

方法：安装opencv软件包，或者把此文件放到根目录

1.用pip安装的opencv不带分类器，所以要下载完整版的，可去官网下载安装，分类器位置在

opencv\build\etc\haarcascades\haarcascade_frontalface_alt_tree.xml
官网地址 https://opencv.org/

2.或者直接下载此文件把它放到根目录就行：下载地址点这里（因为免费的下载比要积分的还麻烦，就要了2分，敬请原谅，渣渣csdn）

一、图片中的人脸检测

代码如下（采用的第2个方法）
'''
import cv2 as cv

# 人脸检测
def face_image():
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
    faces = face_detector.detectMultiScale(gray, 1.02, 5)  # 第二个参数是移动距离，第三个参数是识别度，越大识别读越高
    for x, y, w, h in faces:
        cv.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 后两个参数，一个是颜色，一个是边框宽度
    cv.imshow("result", src)


src = cv.imread("0-common_pics/somePeople.png")
cv.imshow("old", src)
face_image()
cv.waitKey(0)
cv.destroyAllWindows()

'''
二、摄像头中的人脸检测
# 摄像头人脸检测
def face_image(src):
  gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
  face_detector = cv.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
  faces = face_detector.detectMultiScale(gray, 1.02, 5)   # 第二个参数是移动距离，第三个参数是识别度，越大识别读越高
  for x, y, w, h in faces:
    cv.rectangle(src, (x, y), (x+w, y+h), (0, 0, 255), 2)   # 后两个参数，一个是颜色，一个是边框宽度
  cv.imshow("结果", src)


capture = cv.VideoCapture(0)
while(True):
    ret, frame = capture.read()
    frame = cv.flip(frame, 1)
    face_image(frame)
    if cv.waitKey(10) & 0xFF == ord('q'):    # 键盘输入q退出窗口，不按q点击关闭会一直关不掉 也可以设置成其他键。
            break
face_image()
cv.waitKey(0)
cv.destroyAllWindows()

'''
