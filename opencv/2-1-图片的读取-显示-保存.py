import numpy as np
import cv2  # opencv
print(cv2.__version__)  # 3.4.1
'''''
(一)图像的载入
　　　在OpenCv中,加载图像的函数原型如下所示:
     Python:
            cv2.imread(filename[,flags])->retval
            cv.LoadImage(filename iscolor=CV_LOAD_IMAGE_COLOR)->None
     Parameters:
            1--filename---Name of file to loaded
            2--flags------Flags specifying the color type of a loaded image
                     1--CV_LOAD_IMAGE_ANYDEPTH-----if set,return 16-bit/32-bit image when the input has the corresponding depth,
                                                   otherwise convert it to 8-bit
                     2--CV_LOAD_IMAGE_COLOR--------if set,always convert it to the color one
                     3--CV_LOAD_IMAGE_GRAYSCALE----if set,always convert it to the grayscale one
                     1-->0-------------------------return a 3-channel color image
                     2--==0------------------------return a grayscale image
                     3--<0-------------------------return the loaded image as is (with alpha channel)
    Return:
            The function imread loads an image from the specified file and returns it .if the image cannot be read，the function
            returns an empty matrix(Mat::data==NULL)
    Support:
            Currently,the following file formats are supported
                     bmp,jpeg,jpg,pbm,pgm and so on
(二)显示加载进来的图片
　　　在OpenCv中,显示图像的函数原型如下所示:
　　　Ｐython:
            cv2.imshow(winname,mat)--->none
            cv.ShowImage(name,image)-->none
     Parameters:
            1--winname---Name of the window
            2--image-----Image to be shown
     Returns:
            The function imshow displays an image in the specified window.if the window was created wiht the CV_WINDOW_AUTOSIZE flag，
            the image is shown with its original size,Othersize,the image is scaled to fit the window.
(三)创建显示图片的显示窗口
　　在OpenCv中,创建显示函数原型如下所示:
　　Ｐython:
          cv2.namedWindow(winname[,flags])--->none
          cv.NamedWindow(name,flags=CV_WINDOW_AUTOSIZE)--->none
    Parameters:
          1--winname---Name of the window in the window caption that may be used as a window identifier
          2--flags----Flags of the window.The supported flags are:
                  1--WINDOW_NORMAL----If this is set,the user can resize the window(no constraint)
                  2--WINDOW_AUTOSIZE
                  3--WINDOW_OPENGL
(四)销毁创建的窗口
　　在OpenCv中,创建显示函数原型如下所示:
　　Python:
         cv2.destoryWindow(winname)--->none
         cv.DestroyWindow(name)------->none
   Parameters:
         1--winname---Name of the window to be destroyed
   Return:
         The function destroyWindow destroys the window with the given name
(五)让创建的窗口,在指定的位置显示
　　在OpenCv中,函数原型如下所示:
　　Ｐython:
         cv2.moveWindow(winname,x,y)
         cv.MoveWindow(name,x,y)
    Parameters:
         1--winname---Window name
         2--x--The new x-coordinate of the window
            y--The new y-coordinate of the window
    Return:
         Moves window to the specified position
(六)输出图像到文件
　　在OpenCv中,函数原型如下所示:
　　Ｐython:
        cv2.imwrite(filename,img[,params])->retval
        cv.SaveImage(filename,image)
    Parameters:
        1--filename---Name of the file
        2--image------Image to be saved
        3--params-----Format-specific save parameters encoded as pairs paramId_1,paramValue_1...
        4--The following parameters are currently supported:
           1--CV_IMWRITE_JPEG_QUALITY----JPEG
           2--CV_IMWRITE_PNG_COMPRSSION--PNG
           3--CV_IMWRITE_PXM_BINARY------PPM,PGM,PBM
    Returns:
        The function imwrite saves the image to the specified file
'''

imagePath = '0-common_pics/common_1.jpg'

'''
srcImg = cv2.imread(imagePath, cv2.CV_LOAD_IMAGE_GRAYSCALE)  报错
AttributeError: module 'cv2' has no attribute 'CV_LOAD_IMAGE_GRAYSCALE'
opencv 新版本 用的是  IMREAD_GRAYSCALE

src_color_Img = cv2.imread(imagePath, cv2.CV_LOAD_IMAGE_COLOR) 报错
AttributeError: module 'cv2' has no attribute 'CV_LOAD_IMAGE_COLOR'
opencv 新版本 用的是  IMREAD_COLOR
'''
# [1]Load an color Image and convert it to grayscale
srcImg = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
# [2]Load an color image and convert it to grayscale
srcImg_2 = cv2.imread(imagePath, 0)

src_color_Img = cv2.imread(imagePath, cv2.IMREAD_COLOR)  # [3]Load an color image

cv2.namedWindow("[srcImg]")  # [4]create a window　
cv2.namedWindow("[srcImg_2]")
cv2.namedWindow("[color_Img]")

cv2.moveWindow("[color_Img]", 700, 10)
cv2.moveWindow("[srcImg]", 100, 10)  # [5]Moves window to the specified position
cv2.moveWindow("[srcImg_2]", 100, 500)

cv2.imshow("[srcImg]", srcImg)  # [6]display an image in the specified window
cv2.imshow("[srcImg_2]", srcImg_2)
cv2.imshow("[color_Img]", src_color_Img)

cv2.imwrite("0-common_pics/grayCat.png", srcImg)
cv2.imwrite("0-common_pics/colorCat.png", src_color_Img)

cv2.waitKey(0)

cv2.destroyWindow("[srcImg]")  # [6]destroy window
cv2.destroyWindow("[srcImg_2]")
cv2.destroyWindow("[color_Img]")