'''
https://blog.csdn.net/gangzhucoll/article/details/78726069
'''
import cv2 as cv

def equalHist(image):
    """直方图均衡化，图像增强的一个方法"""
    #彩色图片转换为灰度图片
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    #直方图均衡化，自动调整图像的对比度，让图像变得清晰
    dst=cv.equalizeHist(gray)
    cv.imshow("equalHist pic_2",dst)
    cv.imwrite('0-common_pics/pic_equalHist.png',dst)

'''
clahe 算法
https://blog.csdn.net/m_buddy/article/details/53365583
'''
def clahe(image):
    """
    局部直方图均衡化
    把整个图像分成许多小块（比如按8*8作为一个小块），
    那么对每个小块进行均衡化。
    这种方法主要对于图像直方图不是那么单一的（比如存在多峰情况）图像比较实用
    """
    # 彩色图片转换为灰度图片
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #cliplimit：灰度值
    #tilegridsize：图像切割成块，每块的大小
    clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    dst=clahe.apply(gray)
    cv.imshow("clahe pic_3", dst)
    cv.imwrite('0-common_pics/pic_clahe.png',dst)

src=cv.imread('0-common_pics/hist_1.png')
cv.imshow("pic_1", src)

src=cv.imread('0-common_pics/hist_1.png')
#图二
equalHist(src)
#图三
clahe(src)

cv.waitKey(0)
'''
从图中的可以看出来，
比较模糊（图一 pic_1），
经过全图的直方图均衡化后，图片变得清晰了（图二 pic_2），
但是图片明显的对比度偏高，利用局部直方图均衡化后（图三 pic_3），图片看起来就比较舒服了

注意：1、全图的直方图均衡化会导致对比度过度增强，所以在一些情况下应使用局部直方图均衡化；
     2、opencv中直方图均衡化都是基于灰度图像的
'''