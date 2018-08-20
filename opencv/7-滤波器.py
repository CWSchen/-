import numpy as np
import cv2  # opencv

'''
Python OpenCV 滤波器 使用
参考和转载：
http://blog.csdn.net/sunny2038/article/details/9155893
https://blog.csdn.net/mokeding/article/details/19218425
https://blog.csdn.net/xiaowei_cqu/article/details/7785365

一：低通滤波器的目标是降低图像的变化率，
比如将第一个像素替换为该像素周围像素的均值。就可平滑并替代那些强度变化明显的区域。

OpenCV使用blur函数：
dst = cv2.blur(image,(5,5));
# dst -- 处理后的图像
# image -- 待平滑处理的图像
#（5，5） -- 低通滤波器的大小
'''

common_pics_path = "0-common_pics/common_1.jpg"
img = cv2.imread(common_pics_path, 0)
result = cv2.blur(img, (5, 5))

cv2.imshow("Origin", img)  # 显示原图像
cv2.imshow("Blur", result)  # 显示低通滤波器处理后的图像

cv2.waitKey(0)
cv2.destroyAllWindows()

'''
BoxFilter 包滤波器，主要功能是：在给定的滑动窗口大小下，对每个窗口内的像素值进行快速相加求和

Haar特征
https://blog.csdn.net/zhuangxiaobin/article/details/25476833
https://blog.csdn.net/xizero00/article/details/46929261

在模式识别领域，Haar特征是非常熟悉的一种图像特征，可应用于许多目标检测的算法中。
与Haar相似，图像的局部矩形内像素的和、平方和、均值、方差等特征也可以用类似Haar特征的计算方法来计算。

这些特征有时会频繁的在某些算法中使用，因此对它的优化势在必行。Boxfilter就是这样一种优化方法，
它可使复杂度为O(MN)的求和，求方差等运算降低到O(1)或近似于O(1)的复杂度，它的缺点是不支持多尺度。

第一个提出Haar特征快速计算方法的是 CVPR(国际计算机视觉与模式识别) 2001年的那篇经典论文
Rapid Object Detection using a Boosted Cascade of Simple Features ，
它提出了 integral image 的概念，这个方法使得图像的局部矩形求和运算的复杂度从O(MN)下降到了O(4)。

它的原理很简单：首先建立一个数组A，宽高与原图像相等，然后对这个数组赋值，
每个元素的值A[i]赋为该点与图像原点所构成的矩形中所有像素的和。
初始化之后，计算某个矩形像素和时可采用如下方法：
矩形的像素和等于A[4] – A[2] – A[3] + A[1]，共4次运算，即O(4)。

Integral Image 极大的提高了Haar特征的计算速度，它的优点在于能够快速计算任意大小的矩形求和运算。

Boxfilter的原理有点类似 Integral Image，且比它还快，但实现步骤比较复杂。
在计算矩形特征之前，Boxfilter与Integral Image都需要对图像进行初始化（即对数组A赋值），
不同于Integral Image, Boxfilter的数组A中的每个元素的值是该像素邻域内的像素和（或像素平方和），
在需要求某个矩形内像素和的时候，直接访问数组中对应的位置就可以了。因此可以看出它的复杂度是O(1)。

Boxfilter初始化过程如下：
1、给一张图像宽高（M,N），确定待求矩形模板的宽高(m,n)，
   如图紫色矩形。图中每个黑色方块代表一个像素，红色方块是假想像素。

2、开辟一段大小为M的数组，记为buff, 用来存储计算过程的中间变量，用红色方块表示

3、将矩形模板（紫色）从左上角（0，0）开始，逐像素向右滑动，到达行末时，矩形移动到下一行的开头（0，1），
如此反复，每移动到一个新位置时，计算矩形内的像素和，保存在数组A中。
以(0,0)位置为例进行说明：先将绿色矩形内的每一列像素求和，结果放在buff内（红色方块），
再对蓝色矩形内的像素求和，结果即为紫色特征矩形内的像素和，把它存放到数组A中，如此便完成了第一次求和运算。

4、每次紫色矩形向右移动时，实际上就是求对应的蓝色矩形的像素和，
此时只要把上一次的求和结果减去蓝色矩形内的第一个红色块，再加上它右面的一个红色块，
就是当前位置的和了，用公式表示 sum[i] = sum[i-1] - buff[x-1] + buff[x+m-1]

5、当紫色矩形移动到行尾时，需要对buff进行更新。因为整个绿色矩形下移了一个像素，
所以对于每个buff[i], 需要加上一个新进来的像素，再减去一个出去的像素，然后便开始新的一行的计算了。

Boxfilter的初始化过程非常快速，每个矩形的计算基本上只需要一加一减两次运算。
从初始化的计算速度上来说，Boxfilter比Integral Image要快一些，大约25%。
在具体求某个矩形特征时，Boxfilter比Integral Image快4倍，
所谓的4倍其实就是从4次加减运算降低到1次，虽然这个优化非常渺小，
但把它放到几层大循环里面，还是能节省一些时间的。对于那些实时跟踪检测算法，
一帧的处理时间要严格在40ms以下，正是这些细小的优化决定了程序的效率，积少成多，聚沙成塔。
'''
result1 = cv2.boxFilter(img, -1, (5, 5))  # 方框滤波：boxFilter

'''
二：GaussianBlur高斯滤波与低通滤波器不同，低通滤波器中每个像素的权重是相同的，即滤波器是线性的。
   而高斯滤波器中像素的权重与其距中心像素的距离成比例。
   在一些需要对一个像素的周围像素给予更多的重视，可通过分配权重来重新计算这些周围点值。
   可通过高斯函数（钟形函数，即喇叭形数）的权重方案来解决。
'''
result = cv2.GaussianBlur(img, (5, 5), 1.5)
'''
三：中值滤波器是非线性滤波器，对消除椒盐现象特别有用，
result = cv2.medianBlur(image,5)
image -- 原图像
5 -- 孔径尺寸，一个大于1的奇数。5 中值滤波器会使用 5 x 5 的范围来计算。
即对像素的中心值及其 5 x 5 邻域组成了一个数值集，对其进行处理计算，当前像素被其中值替换掉。
如某个像素周围有白或黑的像素，这些白或黑色的像素不会被选择作为中值(最大或最小值不用)，而是被替换为邻域值，示例如下：
'''


def salt(img, n):  # 椒盐现象
    for k in range(n):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 2:
            img[j, i] = 255
        elif img.ndim == 3:
            img[j, i, 0] = 255
            img[j, i, 1] = 255
            img[j, i, 2] = 255
    return img


img = cv2.imread(common_pics_path, 0)
result2 = salt(img, 500)
median = cv2.medianBlur(result2, 5)

cv2.imshow("Salt", result2)
cv2.imshow("Median", median)

cv2.waitKey(0)

'''
中值滤波不会处理最大和最小值，所以就不会受到噪声的影响。相反如果直接采用 blur（低通滤波器）进行均值滤波，
则不会区分这些噪声点，滤波后的图像会受到噪声的影响。
中值滤波器的在处理边缘也有优势。但中值滤波器会清除某些区域的纹理（如背景中的树）
'''
