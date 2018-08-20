'''
低版本                         高版本
cv.NamedWindow 改为了          cv.namedWindow
cv.CV_WINDOW_AUTOSIZE         cv.WINDOW_AUTOSIZE
cv.LoadImage                  改用 cv2.imread               读取图像
cv.ShowImage                  改用 cv2.imread               显示图像
cv.SaveImage                  改用 cv2.imwrite              保存图像

cv.CV_WINDOW_AUTOSIZE         cv.WINDOW_AUTOSIZE
cv.CV_WINDOW_AUTOSIZE         cv.WINDOW_AUTOSIZE
cv.WaitKey                    cv.waitKey
cv.PutText                    cv.putText                图片上写文本
cv.CreateImage                已没有创建图像空间了，用 np.zeros创建
cv.setImageROI                不用了，用numpy中的数组切片设置ROI区域
cv.Copy                       不用了
cv.WaitKey                    cv.waitKey
'''
'''
如果imread png图片会报错如下，但不影响程序运行。
libpng warning: iCCP: known incorrect sRGB profile
'''
'''
srcImg = cv2.imread(imagePath, cv2.CV_LOAD_IMAGE_GRAYSCALE)  报错
AttributeError: module 'cv2' has no attribute 'CV_LOAD_IMAGE_GRAYSCALE'
opencv 新版本 用的是  IMREAD_GRAYSCALE

src_color_Img = cv2.imread(imagePath, cv2.CV_LOAD_IMAGE_COLOR) 报错
AttributeError: module 'cv2' has no attribute 'CV_LOAD_IMAGE_COLOR'
opencv 新版本 用的是  IMREAD_COLOR
'''

'''
opencv下imread绝对路径读取图像失败的问题 (2015-07-25 17:05:46)转载▼
分类： opencv
Mat img=imread("E:\knnpic\knn\supermarket\00000_002.jpg");
imshow("游戏原画",img);

如果像上面这样读取，编译可通过，运行出错，原因是 \ 系统认为是转义字符，
所以"E:\knnpic\knn\supermarket\00000_002.jpg"中\k \k \s \0 被认为是转义字符，读取就会出错。

更改办法又两种：
1.\改为\\
"E:\\knnpic\\knn\\supermarket\\00000_002.jpg"

2.\改为/
"E:/knnpic/knn/supermarket/00000_002.jpg"
'''
'''
Opencv--waitKey()函数详解
是在一个给定的时间内(单位ms)等待用户按键触发;如果用户没有按下键,则接续等待(循环)

 2--如下所示: while(1){ if(waitKey(100)==27)break; } 在这个程序中,
 我们告诉OpenCv等待用户触发事件,等待时间为 100ms，如果在这个时间段内,
 用户按下ESC(ASCII码为27),则跳出循环,否则,则跳出循环

 3--如果设置waitKey(0),则表示程序会无限制的等待用户的按键事件
'''