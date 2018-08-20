# coding: utf-8
import numpy as np
import cv2
from matplotlib import pyplot as plt

'''''
第22章： 直方图
直方图含义：可以对图像灰度分布有整体了解,x轴是灰度值（0到255），y轴是图片中具有同一个
灰度值点的数目，直方图是根据灰度图像绘制的、
BINS:如果需要知道两个像素值之间的数目，只需16个值来绘制直方图
需要把原来的256个值等分成16个小组，取每组的总和。每组就成为BIN。
OpenCV中用hitSize表示BINS。

DIMS:表示收集数据的参数数目，只考虑灰度值，因此参数为1
RANGE:要统计的灰度值范围，一般为[0,256]
cv2.calcHist可以统计图像的直方图
cv2.calcHist(images,channels,masks,histSize,ranges[,hist[,accumulate]])
images:原图像集合，是数组，因此用[],channels:通道，需要用[]括起来，
若图像为灰度图像，值为[0]；彩色图像是[0],[1],[2]分别表示对应的通道是B,G,R
mask:掩膜图像，统计整幅图像就设置为None,统计一部分，需要使用它
histSize:BIN的数目，也要[]括起来,range:像素值范围
'''


def myHist(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    print(hist)
    # plt.imshow(hist , cmap="gray")
    #plt.show(hist)
    '''''
    pyplot.hist(x , bins=10,range=None):
    作用：绘制直方图，x表示类似的一维数组（也可以是图像）
    '''
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()

    '''''
    cv2.imshow("histogram",hist)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    '''''
    numpy.histogram(a,bins=10,range=None)
    a:类似数组的输入数据,bins:用于知道两个像素值之间的数目
    range:统计的像素范围
    返回的hist:数组，bin_edges:length(hist)+1
    hist就是一个含有256个元素的数组，数组中的每一个元素表示数组
    对应下标下的像素点个数
    np.bincount(x,weights=Nne,minlengthNone):
    x:一维数组，返回out:输出的包含
    '''
    hist, bins = np.histogram(img.ravel(), 256, [0, 256])
    print("hist:%d" % len(hist))
    print(hist)
    print("bins:%d" % len(bins))
    print(bins)


'''''
使用掩膜，只需要统计局部区域的直方图，将要统计的部分设置成白色，其余部分为黑色，
就构成一副掩模图像，然后把掩模图像传给函数
这里的白色实际上是全1： 即11111111,黑色时0，因为将掩膜图像和原图像进行与运算后，
只有掩膜的部分还保留成原图像的像素值，黑色部分因为全0相与后就消失后变成黑色
'''


def mask_histogram(img):
    mask = np.zeros(img.shape[: 2], np.uint8)
    mask[100:300, 100:400] = 255
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])
    plt.subplot(221), plt.imshow(img, "gray")
    plt.subplot(222), plt.imshow(mask, "gray")
    plt.subplot(223), plt.imshow(masked_img, "gray")
    plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
    plt.show()


'''''
直方图均衡化
一副图像中大多数店都集中在一个像素值范围内会怎样，
如果一副图片很亮，那么像素值都很高，直方图均衡化会把直方图
做横向拉伸。
直方图均衡化是把直方图映射到广泛分布的直方图中
应用：使所有的图片具有相同的亮度条件
'''


def numpy_historgamEqualization(img):
    '''''
    numpy.ndarray.flatten(order='C'):扁平化，复制一个一维的array出来
    C表示按照行的顺序，F表示按照列的顺序
    '''
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    '''''
    numpy.cumsum(a,axis=None,dtype=None ,out=None):
    a:输入数组
    作用：计算每一行或每一列的累积和，相当于分布求和函数，落在<=a的数量
    '''
    #计算累积分布图（不是标准化的）
    cdf = hist.cumsum()
    '''''
    max():最大值,用分布累积值乘以，直方图最大值/分布累加最大值 ?
    标准化的值=cdf/cdf.max( < 1) * hist.max() < 1
    '''
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    '''''
    pyplot.xlime( ( xmin,xmax ) ):设置x轴的限制范围
    '''
    plt.xlim([0, 256])
    '''''
    pyplot.legend():为线段设置图例(其实就是解释说明)
    '''
    plt.legend(("cdf", "histogram"), loc="upper left")
    plt.show()

    '''''
    找到直方图中的最小值(除了0)，并把它用于wiki中的直方图均衡化公式，使用Numpy
    的掩膜数组，对于掩膜数组的所有操作都只对non-masked元素有效
    构建Numpy掩膜数组，cdf为原数组，当数组元素为0时，掩盖(计算时被忽略)
    numpy.ma.masked_equal(x,value,copy=True):
    掩膜一个数组当之等于给定值时
    '''
    cdf_m = np.ma.masked_equal(cdf, 0)
    #真正的归一化
    cdf_m = ( cdf_m - cdf_m.min() ) * 255 / (cdf_m.max() - cdf_m.min() )
    '''''
    numpy.ma.filled
    '''
    cdf = np.ma.filled(cdf_m, 0).astype("uint8")


def mask_filled(img):
    '''''
    numpy.ndarray.flatten(order='C'):扁平化，复制一个一维的array出来
    C表示按照行的顺序，F表示按照列的顺序
    '''
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    '''''
    numpy.cumsum(a,axis=None,dtype=None ,out=None):
    a:输入数组
    作用：计算每一行或每一列的累积和，相当于分布求和函数，落在<=a的数量
    '''
    #计算累积分布图（不是标准化的）
    cdf = hist.cumsum()
    '''''
    找到直方图中的最小值(除了0)，并把它用于wiki中的直方图均衡化公式，使用Numpy
    的掩膜数组，对于掩膜数组的所有操作都只对non-masked元素有效
    构建Numpy掩膜数组，cdf为原数组，当数组元素为0时，掩盖(计算时被忽略)
    numpy.ma.masked_equal(x,value,copy=True):
    掩膜一个数组当之等于给定值时
    '''
    cdf_m = np.ma.masked_equal(cdf, 0)
    #真正的归一化
    cdf_m = ( cdf_m - cdf_m.min() ) * 255 / (cdf_m.max() - cdf_m.min() )
    '''''
    numpy.ma.filled
    '''
    #将变换应用到图片上
    cdf = np.ma.filled(cdf_m, 0).astype("uint8")
    img2 = cdf[img]
    plt.subplot(121), plt.imshow(img, "gray"), plt.title("original image")
    plt.subplot(122), plt.imshow(img, "gray"), plt.title("historgam equalization")
    plt.xlim([0, 256])
    plt.show()


'''''
opencv中直方图均衡化
cv2.euqalizeHist(src[,dst])->dst
作用：将输入图像的直方图进行均衡化
返回经过直方图处理的图像
处理过程:先计算原图像的直方图，然后对直方图进行标准化
然后计算每个灰度值的直方图： Hi'=在0<=j<i上对H(j)累积求和
将H'转换原图像，即查表: dst(x,y)=H'(src(x,y))

'''


def opencv_histogramEqualization(img):
    equ = cv2.equalizeHist(img)
    '''''
    numpy.hstack(tup)
    作用:按照列来水平合并矩阵将一行上所有的元素的行合并在一起
    tup:n维数组的序列,返回值:n维数组
    实例:
    >>> a = np.ones((2,2))
    >>> b = np.eye(2)
    >>> print np.vstack((a,b))
    [[ 1.  1.]
     [ 1.  1.]
     [ 1.  0.]
     [ 0.  1.]]
    >>> print np.hstack((a,b))
    [[ 1.  1.  1.  0.]
     [ 1.  1.  0.  1.]]
    '''
    # 相当于在原图像上叠加均衡化的直方图，从而使得图片亮度基本保持一致，这个是将两个图像放在一起
    res = np.hstack((img, equ))
    cv2.imwrite("res.png", res)
    plt.subplot(121), plt.imshow(img, "gray"), plt.title("original")
    plt.subplot(122), plt.imshow(equ, "gray"), plt.title("opencv historgam equalization")
    plt.show()


'''''
CLAHE:优先对比适应性直方图均衡化
直方图均衡化的缺点：会改变整个图像的对比度，会丢失信息。
原因：图像的直方图并不是集中在某一个区域
解决方法：使用自适应的直方图均衡化，将图像分成很多小块(tiles，默认8*8)
再对每个小块分别进行直方图均衡化，在每一个区域中，直方图会集中在某一个小的区域
中。如果有噪声，噪声会被放大。
为了避免上述情况要使用对比度限制，对每个小块来说，如果直方图中的bin超过对比度
上限，就把其中的像素点均匀分散到其他的bins中，然后进行直方图均衡化。
最后：为了去除每个小块之间人造边界，使用双线性插值，对小块进行缝合。
'''


def CLAHE_test(img):
    '''''
    cv2.createCLAHE([,clipLimit[,tileGridSize]])->retval
    '''
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    plt.subplot(121), plt.imshow(img, "gray"), plt.title("original")
    plt.subplot(122), plt.imshow(cl1, "gray"), plt.title("clahe equalization")
    plt.show()


if __name__ == "__main__":

    common_pics_path = "0-common_pics/common_1.jpg"
    img = cv2.imread(common_pics_path, 0) # 灰度图计算

    # myHist(img)
    #mask_histogram(img)
    #numpy_historgamEqualization(img)
    #mask_filled(img)
    #opencv_histogramEqualization(img)
    CLAHE_test(img)