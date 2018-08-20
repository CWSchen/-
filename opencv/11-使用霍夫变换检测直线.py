'''
https://blog.csdn.net/sunny2038/article/details/9253823
http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html
'''
#coding=utf-8
import cv2
import numpy as np

common_pics_path = "0-common_pics/common_1.jpg"
img = cv2.imread(common_pics_path, 0)

img = cv2.GaussianBlur(img,(3,3),0)
edges = cv2.Canny(img, 50, 150, apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180,118) #这里对最后一个参数使用了经验型的值
result = img.copy()
for line in lines[0]:
    rho = line[0] #第一个元素是距离rho
    theta= line[1] #第二个元素是角度theta
    print (rho,theta  )
    if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)): #垂直直线
                #该直线与第一行的交点
        pt1 = (int(rho/np.cos(theta)),0)
        #该直线与最后一行的焦点
        pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0])
        #绘制一条白线
        cv2.line( result, pt1, pt2, (255))
    else: #水平直线
        # 该直线与第一列的交点
        pt1 = (0,int(rho/np.sin(theta)))
        #该直线与最后一列的交点
        pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
        #绘制一条直线
        cv2.line(result, pt1, pt2, (255), 1)

cv2.imshow('Canny', edges )
cv2.imshow('Result', result)


'''
注意：

在C++中，HoughLines函数得到的结果是一个向量lines，其中的元素是由两个元素组成的子向量(rho, theta)，
所以lines的访问方式类似二维数组。因此，可以以类似：

[cpp] view plain copy
std::vector<cv::Vec2f>::const_iterator it= lines.begin();
float rho= (*it)[0];
float theta= (*it)[1];
这样的方式访问rho和theta。

而在Python中，返回的是一个三维的np.ndarray！。可通过检验HoughLines返回的lines的ndim属性得到。如：

[python] view plain copy
lines = cv2.HoughLines(edges,1,np.pi/180,118)
print lines.ndim
#将得到3
至于为什么是三维的，这和NumPy中ndarray的属性有关（关于NumPy的相关内容，请移步至NumPy简明教程），如果将HoughLines检测到的的结果输出，就一目了然了：
[python] view plain copy
#上面例子中检测到的lines的数据

3 #lines.ndim属性
(1, 5, 2) #lines.shape属性

#lines[0]
[[  4.20000000e+01   2.14675498e+00]
 [  4.50000000e+01   2.14675498e+00]
 [  3.50000000e+01   2.16420817e+00]
 [  1.49000000e+02   1.60570288e+00]
 [  2.24000000e+02   1.74532920e-01]]
===============
#lines本身
[[[  4.20000000e+01   2.14675498e+00]
  [  4.50000000e+01   2.14675498e+00]
  [  3.50000000e+01   2.16420817e+00]
  [  1.49000000e+02   1.60570288e+00]
  [  2.24000000e+02   1.74532920e-01]]]
概率霍夫变换

观察前面的例子得到的结果图片，其中Hough变换看起来就像在图像中查找对齐的边界像素点集合。但这样会在一些情况下导致虚假检测，如像素偶然对齐或多条直线穿过同样的对齐像素造成的多重检测。

要避免这样的问题，并检测图像中分段的直线（而不是贯穿整个图像的直线），就诞生了Hough变化的改进版，即概率Hough变换（Probabilistic Hough）。在OpenCV中用函数cv::HoughLinesP 实现。如下：
'''
img = cv2.imread(common_pics_path)

img = cv2.GaussianBlur(img,(3,3),0)
edges = cv2.Canny(img, 50, 150, apertureSize = 3)
# lines = cv2.HoughLines(edges,1,np.pi/180,118)
result = img.copy()

#经验参数
minLineLength = 200
maxLineGap = 15
lines = cv2.HoughLinesP(edges,1,np.pi/180,80,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('Result', img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()

