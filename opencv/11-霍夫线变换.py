import numpy as np
import cv2  # opencv

'''
Python OpenCV -- 霍夫线变换（十二）
https://blog.csdn.net/mokeding/article/details/19615873
本文参考和转载：
代码程序使用 sunny2038 博客所提供
http://blog.csdn.net/sunny2038/article/details/9253823
http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html

霍夫线变换
  1.霍夫线变换是一种用来寻找直线的方法.
  2.用霍夫线变换之前, 首先要对图像进行边缘检测的处理，即霍夫线变换的直接输入只能是边缘二值图像.

实现：
    1.一条直线在图像二维空间可由两个变量表示. 例如:
        a.在笛卡尔坐标系: 可由参数: (m,b) 斜率和截距表示.
        b.在极坐标系: 可由参数: (r,theta) 极径和极角表示

        对于霍夫变换, 将用 极坐标系 来表示直线. 因此, 直线的表达式可为:
        化简得: r = x \ cos(theta) + y \ sin(theta)

    2.一般来说对于点 (x_{0}, y_{0}), 我们可以将通过这个点的一族直线统一定义为:
    r_{theta} = x_{0} \cdot \cos \theta  + y_{0} \cdot \sin \theta
    这就意味着每一对 (r_{\theta},\theta) 代表一条通过点 (x_{0}, y_{0}) 的直线.

    3.如果对于一个给定点 (x_{0}, y_{0}) 我们在极坐标对极径极角平面绘出所有通过它的直线,
    将得到一条正弦曲线. 例如, 对于给定点 x_{0} = 8 and y_{0} = 6
    我们可以绘出下图 (在平面 \theta - r):只绘出满足下列条件的点 r > 0 and 0< \theta < 2 \pi.

    4.可对图像中所有的点进行上述操作，若两个不同点在操作后得到的曲线在平面 theta - r 相交,则它们通过同一条直线.
    例如，接上面的例子继续对点: x_{1} = 9, y_{1} = 4 和点 x_{2} = 12, y_{2} = 3 绘图, 得到下图:
    这三条曲线在 theta - r 平面相交于点 (0.925, 9.6), 坐标表示的是参数对 (\theta, r) 或者是说点 (x_{0}, y_{0}), 点 (x_{1}, y_{1}) 和点 (x_{2}, y_{2}) 组成的平面内的的直线.

    5. 那么以上的材料要说明什么呢? 这意味着一般来说,
    一条直线能够通过在平面 \theta - r 寻找交于一点的曲线数量来 检测.
    越多曲线交于一点也就意味着这个交点表示的直线由更多的点组成.
    一般来说我们可以通过设置直线上点的 阈值 来定义多少条曲线交于一点我们才认为 检测 到了一条直线.

    6. 这就是霍夫线变换要做的. 它追踪图像中每个点对应曲线间的交点.
    如果交于一点的曲线的数量超过了 阈值, 那么可认为这个交点所代表的参数对 (\theta, r_{\theta}) 在原图像中为一条直线.

    标准霍夫线变换和统计概率霍夫线变换

    a. 标准霍夫线变换
    原理在上面的部分已经说明了. 它能给我们提供一组参数对(\theta, r_{\theta})  的集合来表示检测到的直线

    在OpenCV 中通过函数 HoughLines 来实现

    b. 统计概率霍夫线变换
    这是执行起来效率更高的霍夫线变换. 它输出检测到的直线的端点 (x_{0}, y_{0}, x_{1}, y_{1})
    在OpenCV 中它通过函数 HoughLinesP 来实现

1. 标准霍夫线变换函数为 cv2.HoughLines。
    输入一幅含有点集的二值图（由非0像素表示），其中一些点互相联系组成直线。
    通常这通过如 Canny 算子获得的一幅边缘图像。
    输出 [ float,float ] 形式的 ndarray,其中每个值表示检测到的线（ρ，θ）中浮点值的参数。
'''
# 示例：
common_pics_path = "0-common_pics/common_1.jpg"
img = cv2.imread(common_pics_path, 0)

img = cv2.GaussianBlur(img, (3, 3), 0)
edges = cv2.Canny(img, 50, 150, apertureSize=3)

# (函数参数3和参数4) 通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
#118 --是经过某一点曲线的数量的阈值
lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)  #这里对最后一个参数使用了经验型的值
result = img.copy()
for line in lines[0]:
    rho = line[0]  #第一个元素是距离rho
    theta = line[1]  #第二个元素是角度theta
    print ( '第一个元素是距离rho' , rho)
    print ( '第二个元素是角度theta' , theta)

    if (theta < (np.pi / 4. )) or (theta > (3. * np.pi / 4.0)):  #垂直直线
        #该直线与第一行的交点
        pt1 = (int(rho / np.cos(theta)), 0)
        #该直线与最后一行的焦点
        pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
        #绘制一条白线
        cv2.line(result, pt1, pt2, (255))
    else:  #水平直线
        # 该直线与第一列的交点
        pt1 = (0, int(rho / np.sin(theta)))
        #该直线与最后一列的交点
        pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
        #绘制一条直线
        cv2.line(result, pt1, pt2, (255), 1)

cv2.imshow('Canny', edges)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
注意：在Opencv C++ 版本中,HoughLines 函数得到的结果是一个向量lines,
其中的元素是由两个元素组成的向量（rho , theta），所以 lines 的访问方式类似二维数组

std::vector<cv::Vec2f>::const_iterator it= lines.begin();
float rho= (*it)[0];
float theta= (*it)[1];

 OpenCV Python 版本中，返回的是一个三维的 np.ndarray 。
 可通过检验HoughLines 返回的 lines 的 ndim 属性得到。
'''

lines = cv2.HoughLines(edges,1,np.pi/180,118)
'''
# 输出结果
#lines.ndim属性
(1, 5, 2) #lines.shape属性
#lines[0]
[[  4.20000000e+01   2.14675498e+00]
 [  4.50000000e+01   2.14675498e+00]
 [  3.50000000e+01   2.16420817e+00]
 [  1.49000000e+02   1.60570288e+00]
 [  2.24000000e+02   1.74532920e-01]]
===============
#lines本身
# [[[  4.20000000e+01   2.14675498e+00]
#   [  4.50000000e+01   2.14675498e+00]
#   [  3.50000000e+01   2.16420817e+00]
#   [  1.49000000e+02   1.60570288e+00]
#   [  2.24000000e+02   1.74532920e-01]]]

1.概率霍夫变换

   通过上面的例子可以看出，其中 Hough 变换看起来就像在图像中查找对齐的边界像素点集合。但这样会在一些情况下导致虚假检测，如像素偶然对齐或多条直线

穿过同样的对齐像素造成的多重检测。

  要避免这样的问题，并检测图像中分段的直线（而不是贯穿整个图像的直线），由此出现了 概率 Hough 变换（Probabilistic Hough）

Python 版本中 由 cv2.HoughLinesP 实现：
'''
img = cv2.imread(common_pics_path)

img = cv2.GaussianBlur(img,(3,3),0)
edges = cv2.Canny(img, 50, 150, apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180,118)
result = img.copy()

#经验参数
minLineLength = 200
maxLineGap = 15
lines2 = cv2.HoughLinesP(edges,1,np.pi/180,80,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines2[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()





