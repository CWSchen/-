import numpy as np
import cv2  # opencv

'''
Python OpenCV -- 轮廓检测 (十四)
https://blog.csdn.net/mokeding/article/details/20153325
http://python.jobbole.com/85253/
参考和转载：
http://blog.csdn.net/sunny2038/article/details/12889059

轮廓检测 Canny 之类的边缘检测算法可根据像素间的差异检测出轮廓边界，但它并没有将轮廓作为一个整体
轮廓是构成任何一个形状的边界或外形线。直方图对比和模板匹配根据色彩的分布来进行匹配，
以下包括：轮廓的查找、表达方式、组织方式、绘制、特性、匹配。

可使用OpenCV 自带的 cv2.findContours() #函数来查找检测物体的轮廓
函数接受的参数为二值图，即黑白的（不是灰度图），所以读取的图像要转成灰度的，再转成二值图。

cv2.findContours() 函数原型：  # Contours 轮廓
contours, hierarchy = cv2.findContours(image, mode, method[, contours[, hierarchy[, offset ]]])
参数：
image -- 要查找轮廓的原图像
mode -- 轮廓的检索模式，它有四种模式：
        cv2.RETR_EXTERNAL  表示只检测外轮廓
        cv2.RETR_LIST 检测的轮廓不建立等级关系
        cv2.RETR_CCOMP 建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。
            如果内孔内还有一个连通物体，这个物体的边界也在顶层。
        cv2.RETR_TREE 建立一个等级树结构的轮廓。

method --  轮廓的近似办法：
    cv2.CHAIN_APPROX_NONE 存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max （abs (x1 - x2), abs(y2 - y1) == 1
    cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，如一个矩形轮廓只需4个点来保存轮廓信息
    cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法

返回值：新版本返回三个值，老版本返回两个值
    cv2.findContours() 函数返回两个值：
    contours -- 轮廓本身，它是一个list,list 中每个元素都是图像中的一个轮廓，用Numpy中的ndarray表示 ，
    每个轮廓是一个ndarray,每个ndarray是轮廓上的点的集合。轮廓中并不存储轮廓上所有的点，
    而只存储可以用直线描述轮廓的点的个数，比如一个“正立”的矩形只有4个点元素。

    hierarchy -- 每条轮廓对应的属性。这是一个ndarray 是个可选返回结果，其中的元素个数和轮廓个数相同，
    每个轮廓contours[i] 对应4个hierarchy元素hierarchy[i][0] ~ hierarchy[i][3] ,
    分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数。


轮廓绘制 OpenCV 使用 cv2.drawContours 在图像上绘制轮廓 函数原型：
cv2.drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset ]]]]])
参数：
    image -- 要绘制轮廓的图像
    contours -- 轮廓本身，在Python 中是一个list
    contourIdx -- 指定绘制轮廓 list 中的哪条轮廓，如果是 -1,则绘制其中的所有轮廓。
    color -- 绘制的颜色及级宽度的属性
    thickness -- 绘制轮廓线的宽度，如果是 -1 (cv2.FILLED)，则为填充模式。

使用示例：
'''
common_pics_path = "0-common_pics/common_1.jpg"
img = cv2.imread(common_pics_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转成灰度图像
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 将灰度图像转成二值图像

_, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓

print (type(contours))
print (type(contours[0]))
print (len(contours))    # 输出list 中轮廓的个数
print (len(contours[0])) # 输出第一个轮廓中的元素的个数，

print (type(hierarchy))
print (hierarchy.ndim)
print (hierarchy[0].ndim)
print (hierarchy.shape)

cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

cv2.imshow("img", img)
cv2.waitKey(0)
