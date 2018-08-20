# *-* coding: utf-8 *-*
import cv2
import numpy as np
from matplotlib import pyplot as plt

'''''
第14章：几何变换
https://blog.csdn.net/qingyuanluofeng/article/details/51582142
14。1扩展缩放
    扩展缩放只是改变图像的尺寸大小。cv2.resize()可以实现该功能，图像的尺寸可以手动设置，可以指定
    缩放因子。可以选择使用不同的差值方法。
    缩放时可用cv2.INTER_AREA，扩展时使用cv2.INTER_CUBIC(慢)，cv2.INTER_LINEAR
    默认改变图像尺寸大小的插值方法都是cv2.INTER_LINEAR

    插值：图像放大时，像素也增加，增加过程就是“插值”程序自动选择信息较好的像素作为增加的像素，放大
    图像时，图像看上去平滑。

    常见插值方法：
    参见csdn这篇文章:http://blog.csdn.net/coy_wang/article/details/5027872
    1最邻近元法
    无需计算，将距离待求像素最近的邻像素灰度付给待求像素。设i+u,j+v为待求像素坐标，则
    待求像素灰度的值f(i+u,j+v)是
    u < 0.5,v < 0.5时为(i,j)....
    出现锯齿，计算量小

    2双线性内插法
    利用待求像素四个邻像素的灰度在两个方向上做线性内插
    对于(i,j+v)，f(i,j)到f(i,j+1)的灰度变化为线性关系，则有:
    f(i,j+v) = [ f(i, j+1) - f(i,j) ] * v + f(i,j)
    例如
    f(i,j+1) = [ f(i,j+1) - f(i,j)] * 2 + f(i,j)
    对于(i+1 , j+v)则有
    f(i+1 , j+v) = [ f(i+1,j+1) - f(i+1,j) ] * v + f(i+1,j)
    从f(i,j+v)到f(i+1,j+v)的灰度变化也为线性关系
    f(i+u,j+v) = [ f(i+1,j+v) - f(i,j+v) ] * u + f(i,j+v)
              =  [ ( [ f(i+1,j+1) - f(i+1,j) ] * v + f(i+1,j) ) - ( [ f(i, j+1) - f(i,j) ] * v + f(i,j) ) ] * u +
                 [ f(i, j+1) - f(i,j) ] * v + f(i,j)
              = ...
    本质：利用相邻像素值求取任意增加x或y后像素值对应的线性的灰度值，连续使用两次求得任意位置处的像素的灰度值
    特点：计算量大，但没有灰度不连续的缺点
    具有低通滤波特点(低频信号通过，超过设定临界值的高频信号则被阻隔，用于数据平滑，图像模糊等)

    3三次内插法
    利用三次多项式S(x)求逼近理论上最佳插值函数sin(x)/x，数学表达式为
         { 1 -2|x|*|x| + |x|*|x|*|x|           , 0 <=|x| <1
    S(x)={ 4 - 8|x| + 5|x|*|x| - |x|*|x|*|x|  , 1 <=|x| < 2
         {0  , |x| >= 2
    主要思想：待求像素(x,y)的灰度值由其周围16个灰度值加权内插得到
    用矩阵相乘做
    特点；计算量较大，插值后的图像效果最好
'''

#图像放缩
def resizeImage():
    '''''
    cv2.resize(src,dsize[,dst[,fx[,fy[,interpolation]]]]) -> dst
    src:输入图像，dsize:输出图像的大小，如果为0，dsize=fx*src.cols , fy*src.rows
    fx:水平放缩尺寸，如果为0，size.width/src.cols，fy:垂直放缩尺寸，如果为0，
    interpolation:插值方法，有以下选项可以选择
    INTER_NEAREST:近邻插值
    INTER_LINERA:双线性插值，默认
    INTER_CUBIC:三次内插（4*4的像素邻居）
    INTER_LANCZOS4
    INTER_AREA：
    '''
    img = cv2.imread("opencv_logo3.png")
    res = cv2.resize(img , None , fx=3 , fy=3 , interpolation=cv2.INTER_CUBIC )
    height , width = img.shape[ : 2]
    #res = cv2.resize(img , (2*width , 2*height) , interpolation=cv2.INTER_CUBIC)
    while(1):
        cv2.imshow("res" , res)
        cv2.imshow("img" , img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

'''''
平移:就是将对象换一个位置。如果沿(x,y)方向移动，移动的距离是(tx,ty)，可以以下面
的方式构建移动矩阵:
M=[1 0 tx]
  [0 1 ty]
可以使用Numpy数组构建该矩阵(数据类型是np.float32)，然后把它传给函数cv2.warpAffine()。
cv2.warpAffine(src,M,dsize[,dst[,flags[,borderMode[,borderValue]]]]) -> dst
src:输入图像,dst:输出图像
M:2*3的变换矩阵,dsize:输出图像的大小
dsize:输出图像的(宽度，高度)
flags:插值方法组合和可选择的标记WARP_INVERSE_MAP（逆转换,dst->src）
borderMode:像素外推方法，当borderMode=BORDER_TRANSPARENT时，它表示
与原图像中的异常值所对应的在目标图像中的像素不会被修改
borderValue:在常数边界情况下使用的值，默认为0
dst(x,y)=src(M11x + M12y + M13 , M21x + M22y + M23)

像素外推是指，像素在给定值外推断其值大小的方法
己知: x = 1 2 3
y = 2 4 6
推出这个关系:
y = 2*x
内插：
x = 2.5
y = 2*x = 5
外推:
x = 4;
y = 2*x = 8

平移变换公式
{x=x0+deltaX
{y=y0+deltaY
矩阵变换公式
[x] [1 0 deltaX] [x0]
[y]=[0 1 deltaY] [y0]
[1] [0 0 1     ] [1 ]
逆变换方法是:
[x0] [1 0 -deltaX] [x]
[y0]=[0 1 -deltaY] [y]
[1]  [0 0 1      ] [1]
若移动后大小不变
当多余部分填充为黑色时
平移结果为
f11 f12 f13 f14         0   0   0   0
f21 f22 f23 f24         0   f11 f12 f13
f31 f32 f33 f34  -----> 0   f21 f22 f23
f41 f42 f43 f44         0   f31 f32 f33

当多余部分为白色时
平移结果为
                        255 255 255 255
                ----->  255 f11 f12 f13
                        255 f21 f22 f23
                        255 f31 f32 f33

若移动后图像尺寸变大
平移结果为
                        0   0   0   0   0
                ----->  0   f11 f12 f13 f14
                        0   f21 f22 f23 f24
                        0   f31 f32 f33 f34
                        0   f41 f42 f43 f44
'''
def translation():
    img = cv2.imread("translation.jpg")
    #注意转换矩阵的数据类型是float32
    M = np.float32( [ [1,0,100],[0,1,50] ] )
    #m = np.arrays([ [1,0,100],[0,1,50] ])
    height , width = img.shape[ : 2]
    #注意cv2.warpAffine中图像大小是(宽度，高度)，但是shape返回的是(高度，宽度)
    dst = cv2.warpAffine(img , M , (width , height) )
    cv2.imshow("源图像",img)
    cv2.imshow("平移(100,50)后的图像" , dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''''
旋转：对一个图像旋转角度 C塔，需要用到下面形式的旋转矩阵
下面的8表示字母 C塔
M=[cos8 -sin8]
  [sin8 cos8 ]
Opencv允许在任意地方进行旋转，但是旋转矩阵的形式应该修改为
[a  b   (1-a)center.x - b.center.y]
[-b a   b.center.x+(1-a).center.x ]
其中:
    a=scale * cos8
    b=scale * sin8
旋转矩阵函数: cv2.getRotationMatrix2D(center , angle , scale)->retval
center:源图像中的旋转中心，angle:旋转角度（正值表示逆时针旋转）,scale:旋转后的缩放因子
center=(宽度y，高度x)

'''
def rotate():
    img = cv2.imread("translation.jpg" , 0)
    height , width = img.shape
    M = cv2.getRotationMatrix2D( (width/2 , height/2) , 90 , 1)
    dst = cv2.warpAffine(img , M , (width , height) )
    cv2.imshow("img" , img)
    cv2.imshow("rotation_img" , dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''''
仿射变换:原图中所有的平行线在结果图像中同样平行。从原图像中找到3个点以及他们在输出图像中的位置。
        然后cv2.getAffineTransform会创建2*3的矩阵，传递给cv2.warpAffine
注意：图像中给出的矩阵中的位置[50,100]就按照正常的横坐标=50，纵坐标=100来看就行,
之前之所以纠结是因为shape函数返回的是高度和宽度，与我如何看图并没有关系
plt.subplot
subplot是为了在一张图中放多个子图，pyplot是有状态的对象，包含了当前的图，画图区域
例如p1=plt.subplot(211)表示创建一个2行1列的图，p1为第一个子图，然后在p1上画曲线，
设置标注标题图例，pyploy.title是给当前子图添加标题
plt.show():

'''
def affineTransform_test():
    img = cv2.imread("drawing.png")
    height , width , channels = img.shape
    pts1 = np.float32( [ [50,50] , [200,50] , [50,200] ] )
    pts2 = np.float32( [ [10,100] ,[200,50] , [100,250] ] )
    '''''
    cv2.getAffineTransform(src,dst)
    src:源图像中三个顶点的坐标,dst:目标图像中三个顶点的坐标
    转换公式
    [xi']=map_matrix * [xi]
    [yi']              [yi]
                       [1 ]
    dst(i)=(xi',yi'),src(i)=(xi,yi) , i=0,1,2
    '''
    M = cv2.getAffineTransform(pts1 , pts2)
    dst = cv2.warpAffine(img , M , (width , height))


    '''''
    pyplot.imshow(x,cmap=None,...)
    X:数组或图像,cmap:colormap色彩地图，默认无
    '''
    plt.subplot(121),plt.imshow(img),plt.title("input")
    plt.subplot(122),plt.imshow(dst),plt.title("output")
    plt.show()


'''''
透视变换：
含义：将图片投影到新的平面
原理：需要3*3的变换矩阵，变换前后直线还是直线，在输入图像找4个点，以及
      他们在输出图像上对应的位置，这四个点中任意三个都不能贡献，变换矩阵
      函数cv2.getPerspectiveTransform()构建，然后把这个矩阵传给函数
      cv2.warpPerspective()
cv2.getPerspectiveTransform(src,dst)
src:源图像中四边形的4个顶点集合
dst:目标图像中的4个顶点集合
转换公式是乘以一个3*3的透视转换矩阵

tiXi'               Xi
tiYi'= map_matrix * Yi
ti                  1

cv2.warpPerspective(src, M , dsize[,dst....])
M:3*3的转换矩阵,dsize:输出图像的大小
'''
def perspectiveTransform():
    img = cv2.imread("sudokusmall.png" )
    height , width , channels = img.shape
    pts1 = np.float32( [ [56,65],[368,52],[28,387],[389,390] ] )
    pts2 = np.float32( [ [0,0],[300,0],[0,300],[300,300] ] )
    M = cv2.getPerspectiveTransform(pts1 , pts2)
    dst = cv2.warpPerspective( img , M , (width , height))
    plt.subplot(121),plt.imshow(img),plt.title("input")
    plt.subplot(122),plt.imshow(dst),plt.title("output")
    plt.show()



if __name__ == "__main__":
    #resizeImage()
    #translation()
    #rotate()
    #affineTransform_test()
    perspectiveTransform()
