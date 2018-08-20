# coding: utf-8
import numpy
import cv2
from matplotlib import pyplot as plt
'''''
15章： 图像阈值
https://blog.csdn.net/qingyuanluofeng/article/details/51585586
像素高于阈值时，给像素赋予新值，否则，赋予另外一种颜色。函数是cv2.threshold()
cv2.threshold(src,thresh,maxval,type[,dst])->retval,dst
作用：用于获取二元值的灰度图像
thresh:阈值，maxval:在二元阈值THRESH_BINARY和逆二元阈值THRESH_BINARY_INV中使用的最大值
返回值retval其实就是阈值
type:使用的阈值类型
THRESH_BINARY
dst(x,y)={maxval if src(x,y) > thresh
         {0      otherwise
THRESH_BINARY_INV
dst(x,y)={0 if src(x,y) > thresh
         {maxval      otherwise
THRESH_TRUNC
dst(x,y)={thresh if src(x,y) > thresh
         {src(x,y)      otherwise
THRESH_TOZERO
dst(x,y)={src(x,y) if src(x,y) > thresh
         {0   otherwise
THRESH_TOZERO_INV
dst(x,y)={0 if src(x,y) > thresh
         {src(x,y) otherwise
cv2.THRESH_BINARY,THRESH_BINARY_INV,THRESH_TRUNC,THRESH_TOZERO,THRESH_TOZERO_INV
'''

def imageThreshold(img):
    #这里的返回值ret其实就是阈值
    ret1,thresh_binary = cv2.threshold(img , 127 , 255 , cv2.THRESH_BINARY)
    ret2, thresh_binary_inv = cv2.threshold(img , 127 , 255 , cv2.THRESH_BINARY_INV)
    ret3 , thresh_trunc = cv2.threshold(img , 127 , 255 , cv2.THRESH_TRUNC)
    ret4 , thresh_tozero =cv2.threshold(img , 127 , 255 , cv2.THRESH_TOZERO)
    ret5,  threshold_tozero_inv = cv2.threshold(img , 127 , 255 , cv2.THRESH_TOZERO_INV)
    print("ret1=%d" % ret1)
    print("ret2=%d" % ret2)
    print("ret3=%d" % ret3)
    print("ret4=%d" % ret4)
    print("ret5=%d" % ret5)
    '''''
    xrange返回从0开始的迭代器序列
    numpy.arange([start,]stop,[step,]):返回等差数列的数组
    例如arange(5)表示返回的是array([0,1,2,3,4])
    plt.subplot(231):表示创建一个2行3列的去U，其中当前图像在第一个区域块中
    plt.xticks( arrange(3),('machao','mayan','maxiping')):设置x轴上坐标刻度
    plt.subplot说白了是为了在一个窗口中显示多个图像
    '''
    titleList = ["source image" , "binary_threshold" , "binary_inv_threshold" , "tunc_threshold" , "tozero_threshold" ,"tozero_inv_threshold" ]
    imgList = [img , thresh_binary , thresh_binary_inv , thresh_trunc , thresh_tozero , threshold_tozero_inv]
    #python3.4中range就是原来的xrange
    for i in range(6):
        plt.subplot(2,3,i+1)
        #注意这里展示的时候也用输入图像，plt.imshow(X,cmap=None)，注意这里的色彩地图使用的是灰度图像
        plt.imshow(imgList[i], "gray")
        plt.title(titleList[i])
        #注意这里x坐标是数组，哪怕是空的也要写
        plt.xticks([])
        plt.yticks([])
    plt.show()

'''''
这一章比较重要
自适应阈值：当同一幅图像上不同部分具有不同亮度时，需要自适应阈值
此时的阈值时根据图像上的每一个小区域计算其对应的阈值
优点：在亮度不同的情况下得到更好的结果
'''
def adaptiveThreshild_test(img):
    '''''
    cv2.medianBlur(src , ksize[,dst])->dst
    作用：使用ksize*ksize的终止过滤来模糊图像，每一个通道是独立处理的。
    src:输入的图像，当孔径尺寸为3或5时，图像深度(存储每个像素所用的位数，
    例如每个像素为8bit，那么取值范围0~255)需要为CV_8U。。。
    ksize:孔径尺寸，它必须是大于1的奇数
    中值模糊
    '''
    img = cv2.medianBlur(img , 5)
    ret , th1 = cv2.threshold(img ,127,255,cv2.THRESH_BINARY)
    '''''
    cv2.adaptiveThreshold(src,maxValue,adaptiveMethod , thresholdType,blockSize , C[,dst])->dst
    作用：将灰度图像转换为二值图像
    二值图像是指每个像素只有两种可能的取值，一般用于描述文字或图形
    maxValue:在条件满足时会赋予像素的非零值,
    adaptiveMethod:自适应阈值用到的动态方法，比如ADAPTIVE_THRESH_MEAN_C 或者 ADAPTIVE_THRESH_GAUSSIAN_C
    thresholdType:阈值类型，这里只能是二元阈值或者是逆二元阈值,blockSize:像素另据的个数，为3,5,7等
    C:从平均值或加权平均值中减去的常量
    THRESH_BINARY:
    dst(x,y)={maxValue if src(x,y) > T(x,y)
             {0      otherwise
    其中T(x,y)是用于对每一个像素计算阈值
    对于ADAPTIVE_THRESH_MEAN_C: T(x,y)是对（x,y）的blockSize * blockSize的邻居平均值减去C
        ADAPTIVE_THRESH_GAUSSIAN_C:阈值取值为相邻区域的加权和，权重为一个高斯窗口
    '''
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    titleList = ["source image" , "Global Threshold(127)" , "Adaptive Mean Threshold" , "Adaptive Gaussian Threshold"]
    imgList = [img , th1 , th2 , th3]
    for i in range(4):
        #注意展示的时候是用灰度图像
        plt.subplot(2,2,i+1);plt.imshow(imgList[i],"gray");plt.title(titleList[i])
        #别忘了设置坐标轴刻度
        plt.xticks([]);plt.yticks([])
    plt.show()

'''''
噪声对图像处理影响很大
平稳噪声：统计特征不随时间变化的噪声
非平稳噪声：统计特征随特征变化的噪声
椒盐噪声：噪声出现的位置是随机的
根据噪声幅值的不同，可以分为高斯噪声和瑞利噪声
常见处理噪声有：均值滤波，中值滤波，灰度最小方差均值滤波等
模板是指矩阵，实际是一种卷积运算，
卷积运算：
均值滤波：对待处理的当前像素，选择模板，该模板为其邻近的若干个像素组成，用模板的均值来替代原像素的值的方法。
g(x,y)=1/M Sum( f(x,y) ),其中f属于S
3*3的模板如下表示
        1   1   1
M=1/9   1   1   1
        1   1   1
g= [ f(x-1,y-1) + f(x,y-1) + f(x+1,y-1) + f(x-1,y) + f(x,y) + f(x+1,y) +
     f(x-1,y+1) + f(x,y+1) + f(x+1,y+1) ] / 9
优点：计算速度快，缺点：降低噪声的同时使图像产生模糊，特别是景物的边缘和细节部分

中值滤波：
含义：对模板的像素由小到大进行排序，再用模板的中值代替原像素的值的方法
权系数矩阵模板:
    1 1 1
mid 1 1 1
    1 1 1
g= median [ f(x-1,y-1) + f(x,y-1) + f(x+1,y-1) + f(x-1,y) + f(x,y) + f(x+1,y) +
            f(x-1,y+1) + f(x,y+1) + f(x+1,y+1) ]
优点：抑制效果好，清晰度基本保持，缺点：对高斯噪声的抑制效果不是很好

高斯模糊模板：
把卷积模板中的值换一下，不是全1，换成一组符合高斯分布的数值放在模板里面，比如中间
的数值最大，两边越来越小，实现函数为cv2.GaussianBlur()，需要指定高斯核的高和宽
(奇数)，沿x与y方向的标准差。高斯核可以有效除去图像的高斯噪声，自己构造高斯核的函数
cv2.GaussianKernel()

'''

'''''
Qtsu's二值化
Qtsu二值化：对一副双峰图像自动根据其直方图计算出一个阈值
'''
def Otsu_test(img):
    ret , th1 = cv2.threshold(img , 127 , 255 , cv2.THRESH_BINARY)
    #在阈值处理的时候，可以在原有二元阈值的基础上加上OTSU,注意这里的thresh值为0
    ret2 , th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU )

    #先高斯模糊，然后阈值处理
    '''''
    cv2.GaussianBlur(src,ksize,sigmaX)
    作用：用指定的高斯核来对源图像进行卷积
    ksize:高斯核大小，是包含(和宽度，核高度)的二元组，必须为正的奇数，或者可以为0，
          然后会根据sigma来计算
    sigmaX:高斯核标准偏差
    '''
    blur = cv2.GaussianBlur(img , (5,5) , 0)
    ret3 , th3 = cv2.threshold(blur , 0 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #这个0是什么意思?，0只是占位用的，后面用plt.hist画直方图
    images = [img , 0 , th1,
              img , 0 , th2,
              blur , 0 , th3]
    cv2.imshow("th1" , th1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    titles = ["Origin Noisy Image" , "Histogram" , "Global Threshold(127)",
              "Origin Noisy Image" , "Histogram" , "Otsu Threshold",
              "Gaussian filtered Image" , "Histogram" , "Otsu Threshold"]
    for i in range(3):
        plt.subplot(3,3,i*3+1), plt.imshow(images[i*3] , 'gray')
        plt.title(titles[3*i]) , plt.xticks([]) , plt.yticks([])
        #pyplot.hist(x,bins=10) ,ravel()是展平成数组（一个），应该是横坐标，和最大值
        plt.subplot(3,3,i*3+2) , plt.hist(images[i*3].ravel() , 256)
        plt.title(titles[3*i+1]) , plt.xticks([]) , plt.yticks([])
        plt.subplot(3,3,i*3+3) , plt.imshow(images[i*3+2] , 'gray')
        plt.title(titles[i*3+2]) , plt.xticks([]) , plt.yticks([])
    plt.show()
    cv2.imshow("img" , img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    imagePath = '0-common_pics/common_1.jpg'
    '''传入的必须是灰度图像，才有1个值，如果是rgb有三个值就不能确定了'''
    img = cv2.imread(imagePath , 0)

    #imageThreshold(img)
    #adaptiveThreshild_test(img)
    Otsu_test(img)
