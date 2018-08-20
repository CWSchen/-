
import cv2

'''
基于Python_OpenCv设置图像的ROI区域   # ROI(region of interest)，感兴趣区域

图像ROI区域的定义方式
在图像算法很多应用场合，图像中处理的目标往往位置相对固定，根据先验信息定义图像的ROI区域可以大幅减小算法的复杂度，
提高算法的效率和鲁棒性。根据图像中目标的形状特征不同，ROI区域的定义方式有多种方式：

（1）简单矩形ROI区域
ROI区域是图像中行列各在某一起始范围的矩形区域中，
ROI区域参数的形式可以以(rowStart, rowEnd, colStart, colEnd)、(rowStart, colStart, width, height)等形式给出。
这是最简单的图像ROI定义方式，也是最容易处理的方式，MATLAB等很多编程方式下都支持直接从矩阵数据中截取子矩阵。
很多别的ROI定义方式，也往往采取先通过bounding box的方式得到矩形ROI再进行下一步的处理。

（2）直线框ROI区域
ROI参数由中心直线的起始点以及框的宽度组成， 简单矩形ROI可以看做是其角度方向为零的特殊情况，这种ROI可以用来提取带方向的边缘。

（3）扇形ROI区域
'''
image_path = "0-common_pics/common_1.jpg"
srcImg = cv2.imread(image_path)  # 将图片加载到内存
cv2.namedWindow("[srcImg]", cv2.WINDOW_AUTOSIZE)  # 创建显示窗口
cv2.imshow("[srcImg]", srcImg)  # 在创建的显示窗口中显示加载的图片

'''
#========================================================================================================
#模块说明:
    由于OpenCv中,imread()函数读进来的图片,其本质上就是一个三维的数组,这个NumPy中的三维数组是一致的,所以设置图片的
    ROI区域的问题,就转换成数组的切片问题,在Python中,数组就是一个列表序列,所以使用列表的切片就可以完成ROI区域的设置
#========================================================================================================
'''
'''
# 实例一 ：设置ROI区域
'''
img_roi_y = 200  # [1]设置ROI区域的左上角的起点
img_roi_x = 200
img_roi_height = 100  # [2]设置ROI区域的高度
img_roi_width = 200  # [3]设置ROI区域的宽度

img_roi = srcImg[img_roi_y:(img_roi_y + img_roi_height), \
          img_roi_x:(img_roi_x + img_roi_width)]

cv2.namedWindow("[ROI_Img]", cv2.WINDOW_AUTOSIZE)
cv2.imshow("[ROI_Img]", img_roi)
cv2.imwrite("/home/wei/caffe/examples/myself/image/cat_ROI.jpg", img_roi)
cv2.waitKey(0)
cv2.destroyWindow("[srcImg]")  # [4]注意:这一点和C++编程中,是有区别的,C++中是不需要手动销毁窗口的
cv2.destroyWindow("[ROI_Img]")  # [5]python中,创建的窗口,需要手动的销毁

'''
# 实例二 ： 将图片切分成指定宽高的小图片
'''
image_save_path_head = "0-common_pics/bank-icon-cut/ROI_CUT"
image_save_path_tail = ".jpg"
seq = 1
ROI_CUT_HEIGHT, ROI_CUT_WIDTH = 360, 500
IMG_WIDTH, IMG_HEIGHT = 1000, 540
RANGE_WIDTH_LEN = int(IMG_WIDTH / ROI_CUT_WIDTH)
RANGE_HEIGHT_LEN = int(IMG_HEIGHT / ROI_CUT_HEIGHT)
'''
只要计算好图片的 宽高 和 要切分得尺寸可以整除，就不会报异常。
'''
for i in range(RANGE_HEIGHT_LEN):  # 计算高 - 列向
    for j in range(RANGE_WIDTH_LEN):  # 计算宽 - 行向

        img_roi = srcImg[(i * ROI_CUT_HEIGHT):((i + 1) * ROI_CUT_HEIGHT), \
                  (j * ROI_CUT_WIDTH):((j + 1) * ROI_CUT_WIDTH)]

        cv2.namedWindow("[ROI_Img]", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("[ROI_Img]", img_roi)
        cv2.waitKey(500)
        cv2.destroyWindow("[ROI_Img]")
        image_save_path = "%s%d%s" % (image_save_path_head, seq, image_save_path_tail)
        cv2.imwrite(image_save_path, img_roi)
        seq = seq + 1
cv2.waitKey(0)
cv2.destroyWindow("[srcImg]")

'''
实例三 ：python中ROI区域的设置
图像处理常常需要设置感兴趣区域(ROI,region of interest),
来专注或简化工作过程.也就是从图像中选择一个图像区域,这个区域是图像分析所关注的重点。
我们圈定这个区域,以便进行进一步处理.
而且,使用ROI指定想读入的目标,可以减少处理时间,增加精度,给图像处理带来不小的便利
　　在Ｃ++中定义ROI区域有两种方法:
        1---使用表示矩形的Rect
        2---使用range
        3--OpemCv1.x中的setImageROI()函数
　　在这里,我就不多说了,可以参考OpenCv的官方教程和相应的源码
'''
srcImg = cv2.imread(image_path, cv2.IMREAD_COLOR)  # [1]加载图片
cv2.namedWindow("[srcImg]", cv2.WINDOW_NORMAL)  # [2]创建图片的显示窗口
cv2.moveWindow("[srcImg]", 100, 100)  # [3]让窗口在指定的位置显示
cv2.imshow("[srcImg]", srcImg)  # [4]显示图片
roiImg = srcImg[20:120, 170:270]  # [5]利用numpy中的数组切片设置ROI区域
srcImg[0:100, 0:100] = roiImg  # [6]将设置的ROI区域添加到源图像中
cv2.namedWindow("[ROIImg]", cv2.WINDOW_NORMAL)
cv2.moveWindow("[ROIImg]", 600, 100)
cv2.imshow("[ROIImg]", srcImg)
cv2.waitKey(0)
cv2.destroyWindow("[ROIImg]")  # [7]销毁窗口,Python编程中,最好加上这一句
cv2.destroyWindow("[srcImg]")