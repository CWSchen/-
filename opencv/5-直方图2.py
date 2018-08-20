'''
https://blog.csdn.net/gangzhucoll/article/details/78715208

一、什么是图像直方图

由于其计算代价较小，且具有图像平移、旋转、缩放不变性等众多优点，广泛地应用于图像处理的各个领域，特别是灰度图像的阈值分割、基于颜色的图像检索以及图像分类。


二、应用范围

图像主题内容与背景分离、图像分类、检索等

三、示例

注意：编写代码前需确保 matplotlib 库已安装，如未安装在命令行中输入：pip install matplotlib
'''
import cv2 as cv
from matplotlib import pyplot as plt


def plot(image):
    """简单的图像直方图"""
    plt.hist(image.ravel(),256,[0,256])
    plt.show("直方图")


def image_his(image):
    """
    这里生成的直方图是opencv 对图片
    进行分割、图像检索等所需要的
    """
    color=('blue','green','red')
    for i ,color in enumerate(color):
        hist=cv.calcHist([image],[i],None,[256],[0,256])
        plt.plot(hist,color=color)
        plt.xlim([0,256])
    plt.show()




pic_1 = '0-common_pics/hist_1.png'
src=cv.imread(pic_1)
cv.imshow('def',src)

#图一
plot(src)
#图二
image_his(src)

cv.waitKey(0)
cv.destroyAllWindows()

