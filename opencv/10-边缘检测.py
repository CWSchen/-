import numpy as np
import cv2  # opencv

'''
Python OpenCV -- Canny 边缘检测 （十一）
https://blog.csdn.net/mokeding/article/details/19581515
https://blog.csdn.net/xiaowei_cqu/article/details/7829481
参考和转载：
程序使用的是 sunny2038 的，最后那个链接就是他的博客
http://wiki.opencv.org.cn/index.php/Canny%E8%BE%B9%E7%BC%98%E6%A3%80%E6%B5%8B
http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
http://blog.csdn.net/sunny2038/article/details/9202641

原理，Canny 边缘检测算法 是 John F. Canny 于 1986年开发出来的一个多级边缘检测算法，
也被很多人认为是边缘检测的 最优算法, 最优边缘检测的三个主要评价标准是:

  低错误率: 标识出尽可能多的实际边缘，同时尽可能的减少噪声产生的误报。
  高定位性: 标识出的边缘要与图像中的实际边缘尽可能接近。
  最小响应: 图像中的边缘只能标识一次。

步骤
 1. 消除噪声。 使用高斯平滑滤波器卷积降噪。
 下面显示了一个 size = 5 的高斯内核示例:

 2.计算梯度幅值和方向。 此处，按照Sobel滤波器的步骤:

    a. 运用一对卷积阵列 (分别作用于 x 和 y 方向):
    b.使用下列公式计算梯度幅值和方向:
        梯度方向近似到四个可能角度之一(一般 0, 45, 90, 135)
 3. 非极大值 抑制。 这一步排除非边缘像素， 仅仅保留了一些细线条(候选边缘)。

 4.滞后阈值: 最后一步，Canny 使用了滞后阈值，滞后阈值需要两个阈值(高阈值和低阈值):
     a. 如果某一像素位置的幅值超过 高 阈值, 该像素被保留为边缘像素。
     b. 如果某一像素位置的幅值小于 低 阈值, 该像素被排除。
     c. 如果某一像素位置的幅值在两个阈值之间,该像素仅仅在连接到一个高于 高 阈值的像素时被保留。

    Canny 推荐的 高:低 阈值比在 2:1 到3:1之间。

使用：OpenCV Python  中 Canny 函数原型
edge = cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])
edge  --  函数返回 一副二值图（黑白），其中包含检测出来的边缘
image --   需要处理的原图像，该图像必须为单通道的灰度图
threshold1  --  阈值1
threshold2  --  阈值2

threshold2  是较大的阈值，用于检测图像中明显的边缘，但一般情况下检测的效果不会那么完美，
边缘检测出来是断断续续的。所以这时候用较小的 threshold1

（第一个阈值）用于将这些间断的边缘连接起来。

apertureSize --  Sobel 算子的大小。
L2gradient -- 布尔值，为True ,刚使用更精确的 L2 范数计算（即两个方向的倒数的平方和再开放），
                      False 将使用 L1 范数（直接将两个方向导数的绝对值相加）。

示例1（静态检测）
'''
# help( cv2.Canny )
# common_pics_path = "0-common_pics/common_1.jpg"
# common_pics_path = r'd:/0-python/0-common_dataSet/yhm/0.jpg'
common_pics_path = 'd:/0-python/0-common_dataSet/榆黄菇-图像识别的数据集/NVRDF9BA2-出菇室 (Preset 3) 2017-03-27-18-00-16.jpg'

# 读取车牌图像
img = cv2.imdecode(np.fromfile(common_pics_path, dtype=np.uint8)
                            , cv2.IMREAD_UNCHANGED)
print(img.shape[0])

# img = cv2.imread(common_pics_path , 0)  # Canny只能处理灰度图，所以将读取的图像转成灰度图
print(img)

img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯平滑处理原图像降噪
canny = cv2.Canny(img, 50, 150)  # apertureSize默认为3

cv2.imshow('Canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 示例2（动态检测）
def CannyThreshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
    detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)
    dst = cv2.bitwise_and(img, img, mask=detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('canny demo', dst)


lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('canny demo')

cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, CannyThreshold)

CannyThreshold(0)  # initialization
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
