import numpy as np
import cv2  # opencv

'''
Python OpenCV 直方图 （五）
https://blog.csdn.net/mokeding/article/details/17634953
https://blog.csdn.net/sunny2038/article/details/9097989

calcHist 函数:
cv2.calcHist([images], channels, mask, histSize, ranges[, hist[, accumulate ]]) #返回hist
参数说明：
images  --   图像对像
channels  --  用于计算直方图的通道
Mask  --
histSize  --   表示直方图分成多少份（多少个直方柱）
ranges  --  表示直方图中各个像素的值，[0.0, 256.0]表示直方图能表示像素值从0.0到256的像素
hist   --
accumulate  --  是一个布尔值，表示直方图是否叠加
'''
# help(cv2.calcHist)
common_pics_path = "0-common_pics/common_1.jpg"
# 灰度图计算
image = cv2.imread(common_pics_path, 0)
hist = cv2.calcHist([image],
                    [0],  # 使用的通道
                    None,  # 没有使用mask
                    [256],  #HistSize
                    [0.0, 255.0])  # 直方图柱的范围

'''
彩色图像（多通道）直方图
使用 OpenCV 方法步骤：1.读取并分离各通道  2.计算每个通道的直方图
'''


def calcAndDrawHist(image, color):
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    '''''
    minMaxLoc 寻找矩阵(一维数组当作向量,用Mat定义) 中最小值和最大值的位置
    注意：多通道图像在使用minMaxLoc()函数是不能给出其最大最小值坐标的，因为每个像
    素点其实有多个坐标，所以是不会给出的。因此在编程时，这2个位置应该给NULL。
    '''
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)

    # 创建绘制直方图的的图像，由于值全为0 所以它是黑色的
    histImg = np.zeros([256, 256, 3], np.uint8)

    hpt = int(0.9 * 256)  # 直方图的范围限定在0-255×0.9之间

    for h in range(256):
        intensity = int(hist[h] * hpt / maxVal)  # 计算直方图的最大值再乘以一个系数

        '''''
        绘制线
        histImg --   图像
        (h,256)  --  线段的第一个端点
        (h,256-intensity)  --  线段的第二个端点
        color  --  线段的颜色
        '''
        cv2.line(histImg, (h, 256), (h, 256 - intensity), color)

    return histImg


if __name__ == '__main__':
    img = cv2.imread(common_pics_path)
    b, g, r = cv2.split(img)  # 使用Opencv 自带的分离函数 split

    histImgB = calcAndDrawHist(b, [255, 0, 0])
    histImgG = calcAndDrawHist(g, [0, 255, 0])
    histImgR = calcAndDrawHist(r, [0, 0, 255])

    cv2.imshow("histImgB", histImgB)
    cv2.imshow("histImgG", histImgG)
    cv2.imshow("histImgR", histImgR)
    cv2.imshow("Img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
在一张图上绘制，这样不用再分离通道，用折线来描绘直方图的边界即可
'''
img = cv2.imread(common_pics_path)
h = np.zeros((256, 256, 3))  # 创建用于绘制直方图的全0图像

bins = np.arange(256).reshape(256, 1)  # 直方图中各bin的顶点位置
color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR三种颜色

'''
对三个通道遍历一次，每次绘制相应通道的直方图的折线
'''
for ch, col in enumerate(color):
    # 计算对应通道的直方图
    originHist = cv2.calcHist([img], [ch], None, [256], [0, 256])

    '''
    OpenCV的归一化函数。该函数将直方图的范围限定在0-255×0.9之间
    '''
    cv2.normalize(originHist, originHist, 0, 255 * 0.9, cv2.NORM_MINMAX)

    '''
    先将生成的原始直方图中的每个元素四舍六入五凑偶取整
   （cv2.calcHist函数得到的是float32类型的数组）
    注意，这里必须使用np.int32(...)进行转换，
    numpy的转换函数可以对数组中的每个元素都进行转换，
    而Python的int(...)只能转换一个元素，如果使用int(...)，
    将导致only length-1 arrays can be converted to Python scalars错误。
    '''
    hist = np.int32(np.around(originHist))

    '''
    将直方图中每个bin的值转成相应的坐标。
    如hist[0] =3，...，hist[126] = 178，...，hist[255] = 5；
    而bins的值为[[0],[1],[2]...,[255]]
    使用np.column_stack将其组合成[0, 3]、[126, 178]、[255, 5]这样的坐标
    作为元素组成的数组。
    '''
    pts = np.column_stack((bins, hist))

    '''
    polylines 根据这些点绘制出折线
    False  --  指出这个折线不需要闭合
    col   --  指定了折线的颜色
    '''
    cv2.polylines(h, [pts], False, col)

'''
反转绘制好的直方图，因为绘制时，[0,0]在图像的左上角
'''
h = np.flipud(h)

cv2.imshow('colorhist', h)
cv2.waitKey(0)

'''
使用 Numpy 直方图计算
NumPy中 histogram 函数应用到一个数组返回一对变量：直方图数组 和 箱式向量。
注意：matplotlib也有一个用来建立直方图的函数(叫作hist,正如matlab中一样)与NumPy中的不同。
主要的差别是pylab.hist自动绘制直方图，而numpy.histogram仅仅产生数据。
'''
img = cv2.imread(common_pics_path)
h = np.zeros((300, 256, 3))
bins = np.arange(257)
bin = bins[0:-1]
color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

for ch, col in enumerate(color):
    item = img[:, :, ch]
    N, bins = np.histogram(item, bins)
    v = N.max()
    N = np.int32(np.around((N * 255) / v))
    N = N.reshape(256, 1)
    pts = np.column_stack((bin, N))
    cv2.polylines(h, [pts], False, col)

h = np.flipud(h)

cv2.imshow('img', h)
cv2.waitKey(0)

'''
直方图
    1、可用来描述不同的参数和事物,如物体的色彩分布,物体的边缘梯度模版以及目标位置的当前假设的概率分布.
    2、是对数据进行统计的一种方法,且将统计值定义到一系列定义好的bin(组距)中,获得一张数据分布的统计图.

比如,现在有一个一维数组,其值从0-255,可以20为组距,分别统计数组中0-20的数据总量,20-40的数据总量,
最后,以这个bin为横轴,统计值作为y轴,得到一张统计图,就是数据范围的直方图,
再比如,一张灰度图像,值也是0-255,也可如下操作得到一张灰度的统计图
(实际上前面所说的直方图均衡化,第一步就是做的这个工作)

图像直方图,是表示数字图像中亮度的直方图,标识了图像中每个亮度值的像素数量,
计算机领域中,常借助于图像的直方图来实现图像的二值化.

总结来说两点
    1.直方图正对图像来说,是图像中像素强度分布的图形表达方式
    2.统计的是每一个强度值所具有的像素个数.

直方图并不局限于统计像素灰度,也可统计图像的任何特征,如梯度,方向等,

术语
    1.dims 需统计的特征的数量,仅统计灰度,dims = 1
    2.bin 每个特征空间中子区段的数目,也叫直条或者组距.
    3.range 每个特征空间的取值范围,如灰度特征空间,取值就是0-255

一.直方图计算

API:void calcHist(
        源图或者元数据指针,
        int 输入的源的个数,
        需要统计的通道索引dims,
        inputarray 可选的操作掩码,
        outputarray 输出目标直方图,
        int 需计算的直方图维度,
        int* 直方图每个维度的尺寸构成的数组,
        float** 每个维度的取值范围构成的数组的指针,
        bool 直方图是否均匀,
        bool 累计标识符
    )

注:源数组中每一个元素的深度和尺寸应该相同,如果操作掩码不为noarray(),
则操作掩码必须为八位,而且和源数组中的每一个元素的尺寸相同,
掩码中不为0区域的坐标对应的源数组中的相应坐标的元素才会被统计,
直方图是否均匀,默认取值为true(均匀),累计标识符默认为false,
该标识符的作用是允许从多个阵列中计算直方图,或者在特定的时间更新直方图.

API:Point* minMaxLoc(
        inputarray 输入数组,
        最小值指针,
        最大值指针,
        point* 最小元素对应的坐标,
        point* 最大元素对应的坐标,
        inputarray 子阵列可选掩码,
    )

注:MatND是直方图对应的数据类型,用来存储直方图.

就这么说显得不知道干嘛的,那让我们看看例子怎么用直方图

二.直方图的匹配
虽然直方图是一个统计值,但有时候也需要比较两个直方图的相似度,作为判定依据的一部分,这时就需要用到直方图的匹配了.

API: double compareHist(源直方图1,源直方图2,int 直方图匹配方法).
注:该API返回值就是匹配的结果,匹配方法有四种
    CV_COMP_CHISQ卡方,返回值越小匹配度越高
    CV_COMP_CORREL相关性匹配,返回值越大匹配程度越高
    CV_COMP_INTERSECT 直方图相交,返回值越大匹配度越高
    CV_COMP_BHATTACHARYYA返回值越小匹配度越高.


三.反向投影

反向投影是一种首先寻找某一特征的直方图模型,然后根据这个模型去寻找图像中是否存在这个特征的解决方案.
反向投影储存的亮度值,代表测试图像中该像素属于某个特征的概率,
也就是说,亮度值相同的位置,属于同一个特征的概率越大,亮起的地方概率更大,内部和边缘之间的阴影影响了检测的精度.

反向投影的作用是在输入图像中寻找特定图像中最匹配的点或者区域,也就是定位模版图像在输入图像的位置.
投影的结果以每个输入图像像素为起点的直方图对比结果,可以看作是单通道浮点型图像,或者是一个二维的概率数组集合.
API:void calcBackProject(
        mat* 输入图像数组指针,
        int 图像数组个数,
        int*需要统计的通道索引,
        inputarray 输入直方图,
        outputarray 目标反向投影阵列,
        float** 输入数组的每一维的边界阵列,
        int 缩放因子,
        bool 直方图是否均匀
    )

注:该函数用来计算反向投影
有时计算复杂图像的反向投影时需抽取出图像的某个通道单独使用,这时就涉及图像的通道复制,
从输入图像中复制某个通道到输出图像的指定通道中.

API:mixChannels(
        mat* 输入图像数组,
        Size_t 输入数组数量,
        Mat*输出数组,
        size_t 输出图像数量,
        const int * 指定复制通道索引数组,
        Size 通达索引的数量
    )

注:该函数属于splite 和mege的高阶通用版本.


四.模版匹配

从一幅图像中寻找和模版最相似的部分的技术,叫做模版匹配,不是基于直方图的匹配技术,
而是通过在输入图像上滑动图像,对实际的图像块和输入图像进行匹配的一种匹配方法.

API:double matchTemplate(输入图像,模版图像,匹配结果的映射图像,指定的匹配方法)

注:输入图像为八位图像或者三十二位浮点型图像,模版和输入图像的类型一致,大小一般不一致,
但是不能大于输入图像,比较结果的映射图像,必然是单通道32位浮点型图像,尺寸为src1.size-temple.size

匹配方法有以下选择 TM_SQDIFF平方差匹配法,最好的匹配是0,匹配结果越差,结果越大,
TM_SQDIFF_NORMED归一化平方差匹配,最差匹配是1,最好匹配是0.TM_CCORR相关匹配0是最坏结果,
结果越大匹配效果越好,TM_CCORR_NORMED归一化相关匹配,1完美匹配,0最坏结果,TM_CCOEFF系数匹配,.

匹配时,对于不同类型的图像,可以使用不同的方法看看哪一种的匹配结果最好,使用的例程代码如下
'''