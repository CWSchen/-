import numpy as np
import cv2  # opencv

'''
python OpenCV 图像像素访问
几点注意说明：
1. 与 c++ 版的opencv 不同，在Python 中灰度图的 img.ndim = 2 ,
而 c++ 中灰度图图像的通道数 img.channel() = 1
2. 关于使用Numpy 的随机数 np.random.random()，Python自带也有一个随机数。
np.random模块中拥有更多的方法，而Python 自带的random只是一个轻量级的模块。
要注意的是 np.random.seed() 不是线程安全的，

而 Python 自带的 random.seed()
import random
random.seed() 是线程安全的。

如果使用随机数时需要用到多线程，建议使用Python自带的random() 和 random.seed() ，
或者构建一个本地的np.random.Random 类的实例

参考和转载地址：http://blog.csdn.net/sunny2038/article/details/9080047
'''

imagePath = '0-common_pics/common_1.jpg'

# 椒盐白点
def salt(img, n): # salt 盐；风趣，刺激性
    for k in range(n):
        i = int(np.random.random() * img.shape[1])  # img.shape[1] 取得img 的行（图片的宽）
        j = int(np.random.random() * img.shape[0])  # img.shape[0] 取得img 的列（图片的高）

        # 判断是否为2维数组(即为灰度图像)
        if img.ndim == 2:
            #像素的访问和访问numpy中ndarray的方法完全一样
            img[j, i] = 255  #设置值为白点 ，灰度图访问;j,i 分别表示图像的行和列

        elif img.ndim == 3:  #判断是否为3维数组(即为RGB图像)
            #opencv 是 BGR 图像访问顺序
            img[j, i, 0] = 255  # 0 -- 为通道，指B通道
            img[j, i, 1] = 255  # 1 -- 为通道，指G通道
            img[j, i, 2] = 255  # 2 -- 为通道，指R通道
    return img


if __name__ == '__main__':
    img = cv2.imread(imagePath)
    saltImage = salt(img, 500)

    cv2.imshow("Salt", saltImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()