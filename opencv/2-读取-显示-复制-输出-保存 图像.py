import numpy as np
import cv2  # opencv

#读取和显示图像
imagePath = '0-common_pics/common_1.jpg'
img = cv2.imread(imagePath) #读取本地图片，目前OpevCV支持bmp、jpg、png、tiff
# cv2.namedWindow("Image")  创建一个窗口用来显示图片
'''
创建图像，新的OpenCV中没有CreateImage接口，即没有cv2.CreateImage函数。如要创建图像，需使用Numpy的函数。
所以图像使用Numpy数组的属性来表示图像的尺寸和通道信息。如输出img.shape得到（500,375,3）,3表示是RGB图像。
'''
emptyImage = np.zeros(img.shape, np.uint8) #根据图像的大小来创建一个图像对象
emptyImage2 = img.copy()  #复制图像
#还可以用 cvtColor 获得原图像的副本
emptyImage3=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 将其转成空白的黑色图像
#emptyImage3[...]=0  # 将其转成空白的黑色图像

cv2.imshow("Image", img) #imshow 显示图片
cv2.imshow("EmptyImage", emptyImage)
cv2.imshow("EmptyImage2", emptyImage2)
cv2.imshow("EmptyImage3", emptyImage3)
'''
保存图像
cv2.imwrite("D:\est.jpg",img,[int(cv2.IMWRITE_JPEG_QUALITY),5])
    第一个参数是保存文件名和路径，第二个是图像矩阵。imwrite() 有个可选的第三个参数
    5 -- 是第三个参数，它针对特定的格式：对于JPEG，其表示的是图像的质量，用0 - 100的整数表示，默认95
    注意: cv2.IMWRITE_JPEG_QUALITY 类型为 long ,必须转换成 int,对于png ,第三个参数表示的是压缩级别。
    cv2.IMWRITE_PNG_COMPRESSION, 从0到9 压缩级别越高图像越小。默认为3.
例如：
    cv2.imwrite("./cat.png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.imwrite("./cat2.png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
'''
cv2.imwrite("0-common_pics/cat2.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
cv2.imwrite("0-common_pics/cat3.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
cv2.imwrite("0-common_pics/cat.png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
cv2.imwrite("0-common_pics/cat2.png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
cv2.waitKey (0)  #等待输入,这里主要让图片持续显示。
cv2.destroyAllWindows()  #释放窗口