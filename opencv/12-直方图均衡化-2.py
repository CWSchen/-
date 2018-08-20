import numpy as np
import cv2  # opencv
'''
直方图均衡化（python实现） https://blog.csdn.net/maryhuan/article/details/12835859
用途：
通常用来增加许多图像的全局对比度，尤其是当图像的有用数据的对比度相当接近的时候。通过这种方法，亮度可以更好地在直方图上分布。这样就可以用于增强局部的对比度而不影响整体的对比度，直方图均衡化通过有效地扩展常用的亮度来实现这种功能。
这种方法对于背景和前景都太亮或者太暗的图像非常有用，这种方法尤其是可以带来X光图像中更好的骨骼结构显示以及曝光过度或者曝光不足照片中更好的细节。

优点：它是一个相当直观的技术并且是可逆操作，如果已知均衡化函数，那么就可以恢复原始的直方图，并且计算量也不大。

缺点：它对处理的数据不加选择，它可能会增加背景噪声的对比度并且降低有用信号的对比度。
'''
'''''
1.均衡化前的直方图和累计直方图
2.均衡化后的直方图和累计直方图
均衡化的函数用的是opencv中的equalizaHist
计算直方图的函数用的是opencv中的calcHist
'''

common_pics_path = "0-common_pics/common_1.jpg"
def drawHist(hist):
    img = np.zeros((256,256),np.uint8) # 创建空矩阵
    r = max(hist)/255
    for i in range (0,256):
        hist[i] = hist[i]/r
        cv2.line(img,(i,255),(i,255-hist[i]),255)
    return img

img = cv2.imread(common_pics_path, 0)
hist1= cv2.calcHist([img], #计算图像的直方图
                    [0], #使用的通道
                    None, #没有使用mask
                    [256], #it is a 1D histogram
                    [0.0,255.0])
hist11 = hist1.cumsum()#累计直方图,求累计值不会改变原数组的值
hist111 = hist11.reshape(hist1.shape)#reshape也不会改变原数组的值
#hist1是二维(ndim),hist11是一维

############均衡化后的投影值和累计值#######
equ = cv2.equalizeHist (img)#均衡化
hist2= cv2.calcHist([equ], #计算图像的直方图
                    [0], #使用的通道
                    None, #没有使用mask
                    [256], #it is a 1D histogram
                    [0.0,255.0])
hist22 = hist2.cumsum()#累计直方图,求累计值不会改变原数组的值
hist222 = hist22.reshape(hist2.shape)#reshape也不会改变原数组的值

a = drawHist(hist1)
a1 = drawHist(hist111)
cv2.imshow("均衡化前的直方图",a)
cv2.imshow("均衡化前的累计直方图",a1)

b = drawHist(hist2)
b1 = drawHist(hist222)
cv2.imshow("均衡化后的直方图",b)
cv2.imshow("均衡化后的累计直方图",b1)

cv2.waitKey(0)
cv2.destroyAllWindows()