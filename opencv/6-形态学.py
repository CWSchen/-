import numpy as np
import cv2  # opencv
'''
形态学的操作包括：腐蚀、膨胀、细化、开运算、闭运算

提示：sunny2038 在博客中提出OpenCV 中函数参数中使用的坐标系和 NumPy 的 ndarray 的坐标系是不同的，
本文参考和转载：
http://blog.csdn.net/sunny2038/article/details/9137759
http://blog.csdn.net/rocky_shared_image/article/details/7821823
http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/opening_closing_hats/opening_closing_hats.html

数字图像处理中的形态学处理是指将数字形态学作为工具从图像中提取对于表达和描绘区域形状有用处的图像分量，

比如：边界、骨架、以及凸壳，还包括用于预处理或后处理的形态学过滤、细化和修剪等。图像形态学处理中主要是

二值图像。

一  基本概念：

1. 二值图像：

   二值图像是每个像素只有两个可能值的数字图像。常用黑白、B&W、单色图像表示二值图像，但是也可以用来表示

每个像素只有一个采样值的任何图像，例如灰度图像等。

   二值图像经常出现在数字图像处理中作为图像掩码或者在图像分割、二值化和dithering的结果中出现。二值图像经常

使用位图格式存储。所以可以解释为二维整数格 Z2 ,


1. 二值图像的逻辑运算

      在图像处理中主要用到的逻辑运算有： 与、或、非（求补）。

2. 膨胀

  膨胀和腐蚀是形态学处理的基础。

  是以得到B的相对与它自身原点的映像并且由Z对映像进行移位为基础的。A被B膨胀是所有位移Z的集合，这样，和A至少

有一个元素是重叠的。

过程：1. 用结构元素B,扫描图像A的每一个像素
2. 用结构元素与其覆盖的二值图像做 “与” 操作
3. 如果都为0，结果图像的该像素为0;否则为1

3. 腐蚀
    对Z中的集合A和B,B对A进行腐蚀的整个过程如下：
     1. 用结构元素B,扫描图像A的每一个像素
     2. 用结构元素与其覆盖的二值图像做 “与”操作
     3. 如果都为1，结果图像的该像素为1. 否则为0

     腐蚀处理的结果是使原来的二值图像减小一圈

4. 匹配

5. 开闭运算
  开运算：是先腐蚀，后膨胀处理。
  闭运算：是先膨胀，后腐蚀处理。

6. 细化
图像细化一般作为一种图像预处理技术出现，目的是提取源图像的骨架,即将原图像中线条宽度大于1个像素的线条细化成
只有一个像素宽，形成“骨架”，形成骨架后能比较容易的分析图像，如提取图像的特征。
细化基本思想是“层层剥夺”，即从线条边缘开始一层一层向里剥夺，直到线条剩下一个像素的为止：图像细化大大地压缩了

原始图像的数据量，并保持其形状的基本拓扑结构不变，
细化算法应满足以下条件：
  1.） 将条形区域变成一条薄线;
  2.）薄线应位于原条形区域的中心：
  3.） 薄线应保持原图像的拓扑特性.
细化分成串行细化和并行细化，串行细化即是一边检测满足细化条件的点，一边删除细化点；并行细化即是检测细化点的时候
不进行点的删除只进行标记，而在检测完整幅图像后一次性去除要细化的点。
常用的图像细化算法有 hilditch 算法， pavlidis 算法和 rosenfeld 算法等。
注： 进行细化算法前要先对图像进行二值化，即图像中只包含 （黑）和 （白）两种颜色。

二： 定义结构元素
形态学处理的核心就是定义结构元素，在 Opencv python 中，使用自带的 getStructuringElement 函数，也可以使用

Numpy 的 ndarray 来定义一个结构元素。

1. 使用getStructuringElement 定义一个结构元素。
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))   # MORPH_CROSS 十字形结构
这里定义了一个 5 * 5 的十字形结构元素

2. 使用NumPy 的 ndarray 来定义结构元素
element = np.uint8(np.zeros((5,5)))   #getStructuringElement 与 Numpy 定义的元素结构是完全一样的
[[0 0 1 0 0]
 [0 0 1 0 0]
 [1 1 1 1 1]
 [0 0 1 0 0]
 [0 0 1 0 0]]


三腐蚀 膨胀 [效果示例代码]
'''
common_pics_path = "0-common_pics/common_1.jpg"
img = cv2.imread(common_pics_path,0) # 参数 0 得到灰度图
#OpenCV定义的结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
#显示原始二值图像
cv2.imshow("original",img)
#腐蚀图像
eroded = cv2.erode(img,kernel)
#显示腐蚀后的图像
cv2.imshow("Eroded Image",eroded)

#膨胀图像
dilated = cv2.dilate(img,kernel)
#显示膨胀后的图像
cv2.imshow("Dilated Image",dilated)
#原图像
cv2.imshow("Origin", img)

#NumPy定义的结构元素
NpKernel = np.uint8(np.ones((3,3)))
Nperoded = cv2.erode(img,NpKernel)
#显示腐蚀后的图像
cv2.imshow("Eroded by NumPy kernel",Nperoded)

cv2.waitKey(0)
cv2.destroyAllWindows()

'''
注意：腐蚀和膨胀，表面看上去好像是一对互逆的操作，实际上，这两种操作不具有互逆的关系。

四： 开运算 闭运算
开运算和闭运算正是依据腐蚀和膨胀的不可逆性演变而来。
开运算和闭运算是将腐蚀和膨胀照一定的次序进行处理，两者也是不可逆的，即先开后闭运算并不能得到原先的图像。
使用示例代码：
'''
img = cv2.imread(common_pics_path,0)
#定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
#闭运算
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#显示腐蚀后的图像
cv2.imshow("Close",closed)

#开运算
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#显示腐蚀后的图像
cv2.imshow("Open", opened)

cv2.waitKey(0)
cv2.destroyAllWindows()

'''
闭运算用来连接被误分为许多小块的对象，而开运算用于移除由图像噪音形成的斑点。因此，某些情况下可以连续运用这两种运算。
如对一副二值图连续使用闭运算和开运算，将获得图像中的主要对象。同样，如果想消除图像中的噪声（图像中的小点），也可以对
图像先用开运算后用闭运算，不过这样也会消除一些破碎的对象。

五：用形态学运算检测边和角点
这里只作为介绍使用，实际使用时使用 Canny 或 Harris 等算法。
1.)检测边缘
膨胀时，图像中的物体会向周围“扩张”；腐蚀时，图像中的物体会“收缩”。由于变化区域只发生在边缘，所以这时将两幅图像相减，
得到的就是图像中物体的边缘。
示例：
'''
image = cv2.imread(common_pics_path,0)
#构造一个3×3的结构元素
element = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
dilate = cv2.dilate(image, element)
erode = cv2.erode(image, element)

#将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像
result = cv2.absdiff(dilate,erode)

#上面得到的结果是灰度图，将其二值化以便更清楚的观察结果
retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY)
#反色，即对二值图每个像素取反
result = cv2.bitwise_not(result)
#显示图像
cv2.imshow("result",result)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
2.）检测拐角
第一步：与边缘检测不同，拐角的检测过程有些复杂，但原理相同，不同的是先用十字形的结构元素膨胀像素，
这种情况下只会在边缘处“扩张”，角点不发生变化。接着用菱形的结构元素腐蚀原图像，导致只有在拐角
处才会“收缩”，而直线边缘都发生变化。

第二步：用X形膨胀原图像，角点膨胀的比边要多。这样第二次用方块腐蚀时，角点恢复原状，而边要腐蚀
的更多，所以当两幅图像相减时，只保留了拐角处。
示例：
'''
image = cv2.imread(common_pics_path, 0)
origin = cv2.imread(common_pics_path)
#构造5×5的结构元素，分别为十字形、菱形、方形和X型
cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(5, 5))
#菱形结构元素的定义稍麻烦一些
diamond = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
diamond[0, 0] = 0
diamond[0, 1] = 0
diamond[1, 0] = 0
diamond[4, 4] = 0
diamond[4, 3] = 0
diamond[3, 4] = 0
diamond[4, 0] = 0
diamond[4, 1] = 0
diamond[3, 0] = 0
diamond[0, 3] = 0
diamond[0, 4] = 0
diamond[1, 4] = 0
square = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
x = cv2.getStructuringElement(cv2.MORPH_CROSS,(5, 5))
#使用cross膨胀图像
result1 = cv2.dilate(image,cross)
#使用菱形腐蚀图像
result1 = cv2.erode(result1, diamond)

#使用X膨胀原图像
result2 = cv2.dilate(image, x)
#使用方形腐蚀图像
result2 = cv2.erode(result2,square)

#result = result1.copy()
#将两幅闭运算的图像相减获得角
result = cv2.absdiff(result2, result1)
#使用阈值获得二值图
retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY)

#在原图上用半径为5的圆圈将点标出。
for j in range(result.size):
    y = int( j / result.shape[0] )
    x = int( j % result.shape[0] )

    if result[x, y] == 255:
        cv2.circle(image, (y, x), 5, (255,0,0))

cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
