import numpy as np
import cv2  # opencv
'''
Python OpenCV -- 直方图均衡化（十三）  类似：photoShop里的色阶
https://blog.csdn.net/mokeding/article/details/19783341

本文参考和转载：
http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/histograms/histogram_equalization/histogram_equalization.html#histogram-equalization
http://blog.csdn.net/sunny2038/article/details/9403059

直方图均衡化
直方图是图像中像素强度分布的图形表达方式。它统计了每一个强度值所具有的像素个数。
直方图均衡化是通过拉伸像素强度分布范围来增强图像对比度的一种方法。

通过上图可以看到像素主要集中在中间的一些强度值上。
直方图均衡化要做的就是 拉伸 这个范围（绿圈圈出来的部分） 少有像素分布

其上的 强度值，对其应用均衡化后，得到中间图所示的直方图。

原理：均衡化指是把一个分布（给定的直方图）映射到另一个分布（一个更宽更统一的强度值分布），所以强度值分布会在整个范围内展开。
要想实现均衡化的效果，映射函数应该是一个 累积分布函数 （cdf），对于直方图  H(i)，
它的 累积分布 H^{'}(i) 是：H^{'}(i) = \sum_{0 \le j < i} H(j)

要使用其作为映射函数，必须对最大值为255（或用图像的最大强度值）的累积分布H^{'}(i) 进行归一化。同上例，累积分布函数为：
最后我们使用一个简单的映射过程来获得均衡化后像素的强度值：

在 Opencv Python 实现

1. 拉伸直方图（使用“查询表”方法）
先检测图像非0的最低（imin）强度值和最高（强度值）。将最低值 imin 设为0，最高值 imax 设为255.中间值按 255.0 * (i - imin) / (imax - imin) + 0.5)

的形式设置。

   示例（这是使用sunny2038 提供的示例代码）：
'''

common_pics_path = "0-common_pics/common_1.jpg"
image = cv2.imread(common_pics_path, 0)
lut = np.zeros(256, dtype = image.dtype )#创建空的查找表
hist= cv2.calcHist([image], #计算图像的直方图
    [0], #使用的通道
    None, #没有使用mask
    [256], #it is a 1D histogram
    [0.0,255.0])

minBinNo, maxBinNo = 0, 255

#计算从左起第一个不为0的直方图柱的位置
for binNo, binValue in enumerate(hist):
    if binValue != 0:
        minBinNo = binNo
        break
#计算从右起第一个不为0的直方图柱的位置
for binNo, binValue in enumerate(reversed(hist)):
    if binValue != 0:
        maxBinNo = 255-binNo
        break
print (minBinNo, maxBinNo) # 1 255

#生成查找表
for i,v in enumerate(lut):
    # print (i)   # 0 - 255
    if i < minBinNo:
        lut[i] = 0
    elif i > maxBinNo:
        lut[i] = 255
    else:
        lut[i] = int(255.0*(i-minBinNo)/(maxBinNo-minBinNo)+0.5)

#计算,调用OpenCV cv2.LUT函数,参数 image --  输入图像，lut -- 查找表
result = cv2.LUT(image, lut)
cv2.imshow("Result", result)
cv2.imwrite("0-common_pics/12.jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
2.Python Numpy直方图均衡化

     示例（示例代码使用 sunny2038 博客提供的）
'''
image = cv2.imread(common_pics_path, 0)

lut = np.zeros(256, dtype = image.dtype )#创建空的查找表

hist,bins = np.histogram(image.flatten(),256,[0,256])
cdf = hist.cumsum() #计算累积直方图
cdf_m = np.ma.masked_equal(cdf,0) #除去直方图中的0值
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())#等同于前面介绍的lut[i] = int(255.0 *p[i])公式
cdf = np.ma.filled(cdf_m,0).astype('uint8') #将掩模处理掉的元素补为0

#计算
result2 = cdf[image]
result = cv2.LUT(image, cdf)

cv2.imshow("OpenCVLUT", result)
cv2.imshow("NumPyLUT", result2)
cv2.waitKey(0)
cv2.destroyAllWindows()

