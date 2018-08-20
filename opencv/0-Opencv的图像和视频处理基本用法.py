'''
https://blog.csdn.net/qingyuanluofeng/article/details/51576498
'''

# coding: utf-8

#学习opencv中基本的图像处理操作
import numpy as np
import cv2

#读取图像
def readImage_my():
    #cv2.imread(filename[,flags]),返回图像，作用：加载图像并返回该图像,flags>0:返回3通道颜色，=0:返回灰度图像，<0:返回的图像带有透明度,alpha是灰度通道，记录透明度信息
    img = cv2.imread("0_snap.png" , 0)
    img = cv2.imread("0_snap.png" , 1)
    img = cv2.imread("0_snap.png" , -1)
    cv2.imshow("image" , img)
    #0表示永久等待键盘输入，waitKey()是键盘绑定函数，时间尺度是毫秒级，特定的几毫秒内，如果有键盘输入，函数会返回按键的ASCII码值
    cv2.waitKey(0)
    #删除建立的窗口，删除特定的窗口用cv2.destroyWindow(),参数是想删除的窗口名
    cv2.destroyAllWindows()

#保存图像
def imwrite_test():
    img = cv2.imread("0_snap.png" , 1)
    '''''
    cv2.imwrrite(filename,img[,params])->返回值，参数:filename是文件名称,img是保存的图像，作用：将图像保存成指定格式的文件,注意这里的params是一个数组
    对于JPEG,可以是有质量的保存 CV_IMWRITE_JPEG_QUALITY 从0到100,100表示最高保存质量，默认95
    对于WEBP,                  CV_IMWRITE_WEBP_QUALITY
    对于PNG，可以是压缩级别     CV_IMWRITE_PNG_COMPRESSION:从0到9，越小表示保存的大小越大，压缩时间越少，默认为3
    alpha为0时表示透明，255时表示不透明
    '''
    #注意cv2.IMWRITE_PNG_COMPRESSION类型为Long，必须转换成int
    outimg = cv2.imwrite("0_snap_save_9.png" , img , [ int(cv2.IMWRITE_PNG_COMPRESSION),9 ] )
    outimg = cv2.imwrite("0_snap_save_0.png" , img , [ int(cv2.IMWRITE_PNG_COMPRESSION),0 ] )
    '''''
    cv2.imshow("outimg" , outimg)
    cv2.waitKey(0)
    cv2.destroyWindow("outimg")
    '''

    '''''
    如果你用的是 64 位系统，你需要将  k = cv2.waitKey(0) 这行改成
    k = cv2.waitKey(0)&0xFF
    '''

#修改像素值
def modifyPixel():
    '''''
    可以根据像素的行和列的坐标获取他的像素值。对于BGR图像而言，返回值为B,G,R的值。对灰度图像而言，会返回灰度值
    :return:
    '''
    img = cv2.imread("./roi.jpg")
    #图像可以理解为二维数组，给出行列则得到BGR，注意不是RGB,一个像素是一个三元组
    px = img[100,100]
    print(px)
    #给出img[row , col , index]  ,index=0时，给出蓝色的像素值
    blue = img[100,100,0]
    print(blue)
    green = img[100,100,1]
    print(green)
    red = img[100,100,2]
    red2  = img.item(100,100,2)
    print(red)
    print(red2)
    #给出BGR的三元组来直接修改像素值
    img[100,100] = [255,255,255]
    print(img[100,100])
    #获取每个像素值，可以使用Numpy的array.item()和array.itemset(),返回值是标量(只有大小)，矢量(大小与方向),想获得所有BGR，需要使用array.item()来分割
    img.itemset( (10,10,2) , 100 )
    red3 = img.item(10,10,2)
    print(red3)
    #img.shape返回行数，列数，色彩通道数（灰度图像不返回通道数）
    print(img.shape)


#提取出感兴趣的图像
def imageROI():
    img = cv2.imread("./roi.jpg" , 1)
    #行对应了图像的高度，列数对应了图像的宽度,190,300
    '''''
    我现在比较疑惑的地方就是shape返回的number of rows, columns and channels (if image is color):
    shape返回的是(190, 302, 3)
    实际上该图片是302*190的，也就是宽度是302对应列数
    那么其实shape返回的值可以理解为:高度(y) ， 宽度(x)

    行数->对应图像多少行->图像高度->坐标y->对应于img[, y] ，总结: 行数->y
    列数->对应图像多少列->图像宽度->坐标x->对应于img[x, ]
    '''
    print(img.shape)
    #img.size返回图像的像素数目 = 高度 * 宽度 * 3（色彩通道）
    print(img.size)
    #img.dtype: 返回图像的数据类型
    print("图像的数据类型:")
    print(img.dtype)
    '''''
    分析：如果我要移动足球到垂直上方处，那么首先x是固定不动的
    由于shape返回190,302
    因此宽度是302,高度是190，因此x的范围是[0,302],y的范围是[0,190]
    球在右下角处，x大概是[150,225]
    y大概是[160,190]
    因为移动到正上方，所以x不变，y减小为[10,40]
    现在关键就是对img[]
    shape(y,x)
    img(y,x)
    因此真正的位置应该是
    '''
    ball = img[160:190 , 150:225 ]
    img[10:40,150:225] = ball
    '''''
    ball = img[150:225 , 160:190]
    img[ 150:225 , 10:40 ] = ball
    '''
    cv2.imwrite("roi_modify.jpg" , img , [int(cv2.IMWRITE_JPEG_QUALITY) , 100 ])
    outimg = cv2.imread("roi_modify.jpg" , 1)
    cv2.imshow("roi_out" , outimg)
    cv2.imshow("roi" , img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#拆分及合并图像通道，适用：对BGR三个通道分别进行操作，或者把独立通道的图片合并成BGR图像
def mergeAndSplitImage():
    img = cv2.imread("./roi.jpg" )
    b,g,r = cv2.split(img)
    #img = cv2.merge(b, g ,r)
    print("分离的通道的bgr分别是")
    print(cv2.split(img))
    #合并时候最多接受两个参数
    img2 = cv2.merge(b, g )
    b = img[: , : , 0]
    print("b色彩通道是")
    print(b)
    #使得所有像素红色通道值为0,split耗时，能用Numpy索引就尽量使用
    img[: , : , 2] = 0
    cv2.imshow("image" , img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#图像加法
def imageAdd():
    #cv2.add(src1 , src2 [ ,dst [, mask [,dtype]]]) ，返回dst，作用:对两幅图像进行加法运算
    #opencv中的加法是饱和操作，超过最大值按最大值算，小于最小值按最小值算（例如大于1变成1）
    #numpy中的加法是模操作,超过后，取模
    #这里[250]表示是一个数组，该数组只有一个元素250
    x = np.uint8([250])
    print("x = np.uint8([250])的结果是")
    print(x)
    y = np.uint8([10])
    print("cv2饱和加法结果是")
    print(cv2.add(x,y)) # 250 + 10 = 260 => 255
    print("numpy取模加法结果是")
    print(x+y)
    img1 = cv2.imread("./0_snap.png")
    img2 = cv2.imread("./9_snap.png")
    resImg = cv2.add(img1 , img2)
    cv2.imshow("image add",resImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''''
图像混合，也是加法，只不过给与两幅图像的权重不同，给人混合或者透明的感觉
图像混合的计算公式
g(x) = (1-a)f(x) + ah(x)
通过修改a的值，实现混合
'''
def imageBlending():
    img1 = cv2.imread("./blend1.jpg")
    img2 = cv2.imread("./blend2.jpg")
    '''''
    cv2.addWeighted(src1 , alpha , src2 , beta , gamma[ , dst [,dtype]] )，返回dst,作用
    dst( = saturate( src1 * alpha + src2 * beta + gamma )
    saturate:饱和
    '''
    dst = cv2.addWeighted(img1 , 0.7 , img2 , 0.3 , 0)
    cv2.imshow("dst" , dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#按位运算，操作有:AND , OR , NOT , XOR，作用：选择非矩形ROI操作会很有用
def bitOperation():
    img1 = cv2.imread("roi.jpg")
    img2 = cv2.imread("opencv_logo.jpg")
    #希望把logo放在左上角
    rows , cols , channels = img2.shape
    roi = img1[0 : rows , 0 : cols]

    #现在创建对于logo的掩码:将源码与掩码（需要的字段位为1）经过或运算得到符合需求的结果

    '''''
    cv2.threshold(src, thresh , maxval , type[,dst] )返回retval,dst
    src输入数组或者图像，dst输出图像，maxval用于二元阈值的最大值,type:阈值类型
    作用：将阈值应用到单通道数组，需要用到灰度图像,主要是过滤掉太大或太小的图像
    THRESH_BINARY
    >threshold,为maxval，否则为0
    \texttt{dst} (x,y) =  \fork{\texttt{maxval}}{if $\texttt{src}(x,y) > \texttt{thresh}$}{0}{otherwise}
    THRESH_BINARY_INV
    \texttt{dst} (x,y) =  \fork{0}{if $\texttt{src}(x,y) > \texttt{thresh}$}{\texttt{maxval}}{otherwise}
    THRESH_TRUNC
    \texttt{dst} (x,y) =  \fork{\texttt{threshold}}{if $\texttt{src}(x,y) > \texttt{thresh}$}{\texttt{src}(x,y)}{otherwise}
    THRESH_TOZERO
    \texttt{dst} (x,y) =  \fork{\texttt{src}(x,y)}{if $\texttt{src}(x,y) > \texttt{thresh}$}{0}{otherwise}
    THRESH_TOZERO_INV
    \texttt{dst} (x,y) =  \fork{0}{if $\texttt{src}(x,y) > \texttt{thresh}$}{\texttt{src}(x,y)}{otherwise}
    '''
    img2gray = cv2.cvtColor(img2 , cv2.COLOR_BGR2GRAY)
    ret , mask = cv2.threshold( img2gray , 175 , 255 , cv2.THRESH_BINARY )
    '''''
    cv2.bitwise_not(src[,dst[,mask]]) ,
    src:输入数组,dst:输出数组（与src有同样的大小和类型）,mask:可选择的操作掩码
    作用：按位取反
    dst(I) = 取反src(I)
    bitwise表示按位
    '''
    mask_inv = cv2.bitwise_not(mask)

    #下面就是讲ROI区域进行处理，取roi中与mask中不为零的值对应的像素的值，其他值为0
    #注意这里必须有mask=mask 或者mask=mask_inv，其中的mask= 不能忽略
    '''''
    cv2.bitwise_and(src1 ,src2[,dst[,mask]])->dst
    src1:第一个输入数组或者标量
    src2:第二个数组
    src:单通道的输入数组
    value:标量值
    dst:输出数组
    mask:掩码
    计算按位与
    dst(I) = src1(I) & src2(I) , if mask(I) != 0
    '''
    #这里的roi是足球照片，用于背景,mask是logo的灰度图像，0是黑，255是白，也就是把白色部分的像素拿出来求与，其实就是把足球偏白色的部分拿出来
    img1_bg = cv2.bitwise_and(roi, roi , mask = mask)
    #取roi中与mask_inv中不为0的值对应的像素的值，其他值为0，把logo中黑色部分提取出来
    img2_fg = cv2.bitwise_and(img2 , img2 , mask = mask_inv)

    #将ROI中的logo和修改主要的图像
    dst = cv2.add(img1_bg , img2_fg)
    #替换原来的图像
    img1[0:rows , 0:cols] = dst
    cv2.imshow("res" , img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()








'''''
Canny边缘检测：
原理：1986年由John F.Canny提出的算法
步骤:
1噪声去除
由于边缘检测容易受到噪声影响，因此需去除噪声，第一步是使用5*5的高斯滤波器
（先看5*5高斯滤波器）

2计算图像梯度
对平滑后的图像使用Sobel算子计算水平方向和竖直方向的一阶导数（图像梯度）
(Gx和Gy)。根据得到的这两幅梯度图(Gx和Gy)找到边界的梯度和方向，公式如下:
Edge_Gradient(G)=sqrt( Gx*Gx + Gy*Gy )
Angle( C塔 ) = tan-1(上标)(Gx/Gy)
梯度的方向一般总是与边界垂直。梯度方向有四类：垂直，水平，和两个对角线
（算子和梯度也需要看）

3非极大值抑制
获得梯度的方向和大小后，对整幅图像做扫描，去除非边界上的点。对每个像素检查，
看这个点的梯度是不是周围具有相同梯度方向中的点中最大的

4滞后阈值
需要设置两个阈值：minVal和maxVal。当图像的灰度梯度高于maxVal时被认为是真的
边界，低于minVal的边界会被抛弃。介于两者之间的话，需要看该点是否与某个被确定
为真正的边界点相连，如果是就认为它是边界点，否则就抛弃。
边界是长的线段。
'''





if __name__ == "__main__":
    #readImage_my()
    #imwrite_test()
    #modifyPixel()
    #imageROI()
    #mergeAndSplitImage()
    #imageAdd()
    #imageBlending()
    bitOperation()















'''''
可能有用的一些内容（简要了解，后面需要时再仔细学习）:
物体跟踪:应用提取某特定颜色的物体(从BGR转换到HSV后，HSV更容易表示特定颜色）
步骤:
1从视频中获取每一帧图像
2将图像红钻换到HSV空间
3设置HSV阈值到蓝色范围

学习轮廓后，可以找到物体的重心，根据重心来跟踪物体

几何变换:
1扩展缩放,cv2.resize(),缩放使用cv2.INTER_AREA,扩展时使用v2.INTER_CUBIC(慢)和v2.INTER_LINEAR(快)
默认改变图像尺寸大小插值方法是cv2.INTER_LINEAR

2平移：将对戏那个换一个位置,cv2.warpAffine(宽度，高度，输出图像的大小)
宽度对应列数，高度对应行数

3旋转:cv2.getRotationMatrix2

4仿射变换:原图中所有的平行线在结果图像中同样平行,为了创建这个矩阵需要从原图像
中找到三个点以及他们在输出图像中的位置
cv2.getAffineTransform , cv2.warpAffine

5透视变换
视角变换，需要3*3变换矩阵，在变换前后直线还是直线，要构建这个矩阵，需要在输入图像
中找4个点，以及他们在输出图像上对应的位置，4个点中任意三个不能共线
变换矩阵可通过cv2.getPerspectiveTransform()构建
然后矩阵传递给cv2.warpPerspective

'''''



'''''
图像阈值：
当像素高于阈值时，给这个像素赋予新值
cv2.threshhold()，第一个参数是原图像，原图像应该是灰度图，第二个参数就是用来对像素值
进行分类的阈值。第三个参数是像素高于（或小于）阈值时应被赋予的新像素值

自适应阈值：
根据图像上的每一个小区域计算与其对应的阈值
Adaptive Method

Otsu's二值化
对一个双峰图像自动根据其直方图计算出其阈值
cv2.threshold()，多传入一个参数cv2.THRESH_OTSU，把阈值设为0

'''

'''''
图像平滑(模糊)
使用低通滤波器对图像进行模糊，使用自定义的滤波器对图像进行卷积(2D卷积)

2D卷积:
低通滤波去除噪音，模糊图像，高通滤波找到图像的边缘
cv.filter2D()

图像模糊(图像平滑)
噪音是指高频成分，边界也会被模糊一点
平均：归一化卷积狂完成，用卷积框覆盖区域所有像素平均值来代替中心元素
可以使用函数cv2.blur()和cv2.boxFilter()来完成

高斯模糊：把卷积核换成高斯核，中心值最大，其余会随距离中心元素的距离递减

中值模糊:用卷积框对应像素的中值来代替中心像素的值

双边滤波

'''


'''''
图像梯度:
梯度简单说就是求导
梯度滤波器也叫高通滤波器:Sobel，Scharr（求一阶导数）,Laplacian算子(二阶导数)

'''