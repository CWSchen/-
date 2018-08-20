'''
来自简书：https://www.jianshu.com/p/5b2d7146172d
【解决问题】Python3.x使用 OpenCV无法读取中文路径图像
2018.01.27 01:04*
1 环境介绍
Windows7 x64，PyCharm 2017.3 Community，Python 3.6，OpenCV 3.4。

2 问题描述
在PyCharm环境，使用OpenCV的cv2.imread(filename)方法读取图像。
当filename 中的路径或者图像文件名包含汉字的时候，cv2.imread(filename)读取不到图像数据，导致后续报错：NoneType。

3 示例代码
以下示例代码运行会报错。
import cv2
# 读取车牌图像
test_sample_file="云A526EG.jpg"
image_origin = cv2.imread(test_sample_file)
print(image_origin.shape[0])
报错信息：

Traceback (most recent call last): File "E:/python/test/demo.py", line 25, in print(image.shape[0])
AttributeError: 'NoneType' object has no attribute 'shape'
2.原因分析
opencv的Python版本，cv2.imread(filename)方法不支持中文路径的文件读入。

3.解决方法
使用OpenCV的cv2.imdecode(buf, flags)方法。代码如下：

import cv2
import numpy as np
# 读取车牌图像
test_sample_file="云A526EG.jpg"
image_origin = cv2.imdecode(np.fromfile(test_sample_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
print(image_origin.shape[0])
参考列表：
1.opencv官方文档
https://docs.opencv.org/3.4.0/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56


来自知乎：https://www.zhihu.com/question/47184512
请教各位朋友cv2的python版本中imwrite无法生成带有中文路径的图片？
# python3版本
# imread
path_file = "sample"
img = cv2.imdecode(np.fromfile(path_file,dtype=np.uint8),-1)

#imwrite
_path = "sample"
cv2.imencode('.jpg',img)[1].tofile(_path)
# python 2版本
import cv2
import sys
reload(sys)
sys.setdefaultencoding('u8')
path_file = u"sample"
img = cv2.imread(path_file.decode('u8').encode('gbk'),-1)

由于python3字符串采用utf8编码，cv2.imread将utf8当作本地码(如GBK)，
这个问题无解了。Python 2.7可以用decode/encode的方法
（包括崔鸣浩用的默认GBK编码）解决，此方法在Python 3中已不能用了。
可以采用以下方法代替
#imread
img = cv2.imdecode(np.fromfile('unicode_path', dtype=np.uint8), flags)
# imwrite
cv2.imencode('.jpg', img)[1].tofile('unicode_path')
imencode/imdecode包含了imwrite/imread的参数

谢邀，我测试的环境是opencv2.4.9+python2.7首先说明一点，题主所说的“cv2不支持中文”是不正确的，题主这里的问题应该是python字符编码的问题；运行下面的代码应该是可以得到你想要的结果：import cv2
cap = cv2.VideoCapture("demo_iccv13.avi")
frameList = []
name = "./蓝天/1.jpeg"
while(cap.isOpened()):
    ret , frame = cap.read()
    if ret == True:
        frameList.append(frame)
    else:
        break
outimg = cv2.imwrite(name , frameList[0], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
cap.release()
'''
'''
Python3.x opencv操作中文文件
2017年02月10日 10:24:19
阅读数：2164
我用的是python3.5，本身用file打开中文文件是没有问题的，但是用opencv就不行，网上看到很多解决版本，可能都是针对python2.x的，没有效果，后来在知乎上看到一个解决方法，测试有效，引用在这里：

冯卡门
由于python3字符串采用utf8编码，cv2.imread将utf8当作本地码(如GBK)，这个问题无解了。Python 2.7可以用decode/encode的方法（包括崔鸣浩用的默认GBK编码）解决，此方法在Python 3中已不能用了。
可以采用以下方法代替
imread

img = cv2.imdecode(np.fromfile(‘unicode_path’, dtype=np.uint8), flags)
imwrite

cv2.imencode(‘.jpg’, img)[1].tofile(‘unicode_path’)
imencode/imdecode包含了imwrite/imread的参数
'''
import numpy as np
import cv2  # opencv
# help( cv2.Canny )
# common_pics_path = "0-common_pics/common_1.jpg"
common_pics_path = r'd:/0-python/0-common_dataSet/yhm/0.jpg'
img0 = cv2.imread(common_pics_path)


''' 普通的转码，对中文目录，也不好使 '''
common_pics_path2 = common_pics_path.encode('gbk')
print(common_pics_path2)
common_pics_path3 = common_pics_path2.decode('gbk')
print(common_pics_path3)

'''
读取中文目录
'''
common_pics_path = 'd:/0-python/0-common_dataSet/榆黄菇-图像识别的数据集/NVRDF9BA2-出菇室 (Preset 3) 2017-03-27-18-00-16.jpg'
# 读取车牌图像
img = cv2.imdecode(np.fromfile(common_pics_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
print(img.shape[0])
print(img)

cv2.imshow("[srcImg]",img0)
# cv2.imshow('img',img)

'''
Opencv--waitKey()函数详解
是在一个给定的时间内(单位ms)等待用户按键触发;如果用户没有按下键,则接续等待(循环)

 2--如下所示: while(1){ if(waitKey(100)==27)break; } 在这个程序中,
 我们告诉OpenCv等待用户触发事件,等待时间为 100ms，如果在这个时间段内,
 用户按下ESC(ASCII码为27),则跳出循环,否则,则跳出循环

 3--如果设置waitKey(0),则表示程序会无限制的等待用户的按键事件
'''
if cv2.waitKey(0) == 27:
    cv2.destoryAllWindows()