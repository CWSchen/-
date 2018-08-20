import numpy as np  # [1]导入python中的数值分析,矩阵运算的程序库模块
import cv2  # [2]导入OpenCv程序库模块
from matplotlib import pyplot as plt  # [3]仅仅导入了matplotlib绘图程序库中的一个子模块

'''
(一)OpenCv中,python接口的基本的绘图函数  类似 前端 h5的canvas svg等
　１－用于绘制直线的--------cv2.line()函数
　２－用于绘制椭圆的--------cv2.ellipse()函数
　３－用于绘制矩形的--------cv2.rectangle()函数
　４－用于绘制圆的-----------cv2.circle()函数
　５－用于绘制多边形的----cv2.fillploy()函数
　６－用于绘制文本的------cv2.putText()函数
'''
'''
【模块１】定义一个画直线的函数
    #1--img--------[1]画直线的图片
    #2--start------[2]直线的起点
    #3--end--------[3]直线的终点
    #4--(255,0,0)--[4]直线的颜色
    #5--thickness--[5]直线的粗细
    #6--lineType---[6]直线的类型--实线,虚线　
'''


def DrawLine(srcImg, start, end):
    thickness = 2
    lineType = 8
    cv2.line(srcImg, start, end, (255, 0, 0), thickness, lineType)


'''''【模块２】定义一个画矩形的函数'''
# 1--画一个矩形的时候,只需要知道两点:左上角定点和右下角定点
# 2--画一个矩形,只需要知道两个Point
def DrawRectangle(srcImg, leftTopPoint, rightButtomPoint):
    thickness = 2
    lineType = 8
    cv2.rectangle(srcImg, leftTopPoint, rightButtomPoint, (0, 0, 255), thickness, lineType)


'''''【模块３】画圆'''
#1--画圆,我们只需要知道原型的中心和半径的大小
def DrawCircle(srcImg, centerPoint, radius):
    thickness = 2
    lineType = 8
    cv2.circle(srcImg, centerPoint, radius, (0, 255, 0), thickness, lineType)


'''''【模块４】画椭圆'''


def DrawEllipse(srcImg, centerPoint, radius, angle):
    thickness = 2
    lineType = 8
    cv2.ellipse(srcImg, centerPoint, radius, angle, 0, 360, (255, 255, 0), thickness, lineType)


srcImg = np.zeros((512, 512, 3), np.uint8)  #[1]创建一个width*heigth=512*512的数通道的黑色彩色图片
DrawLine(srcImg, (0, 0), (511, 511))  #[2]画直线
DrawLine(srcImg, (0, 100), (511, 511))

DrawRectangle(srcImg, (10, 10), (300, 300))  #[3]画矩形

DrawCircle(srcImg, (255, 255), 100)  #[4]画圆
DrawCircle(srcImg, (255, 255), 200)

DrawEllipse(srcImg, (255, 255), (150, 20), 0)  #[5]画椭圆,中心点(255,255),长半轴150,短半轴20,旋转角度０
DrawEllipse(srcImg, (255, 255), (150, 20), 45)  #[5]画椭圆,中心点(255,255),长半轴150,短半轴20,旋转角度45
DrawEllipse(srcImg, (100, 255), (150, 20), 135)  #[5]画椭圆,中心点(100,255),长半轴150,短半轴20,旋转角度45
#[6]在图片上绘制字体
cv2.putText(srcImg, "I am Maweifei,OPenCv", (0, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, 8)

cv2.namedWindow("[srcImg]")
cv2.moveWindow("[srcImg]", 10, 10)
cv2.imshow("[srcImg]", srcImg)
cv2.waitKey(0)
cv2.destroyWindow("[srcImg]")