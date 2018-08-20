# coding: utf-8
'''''
第13章主要介绍：颜色空间转换
https://blog.csdn.net/qingyuanluofeng/article/details/51582137
'''

import cv2
import numpy as np

'''''
经常用到的颜色空间转换是: BGR<->Gray 和 BGR<->HSV
cv2.cvtColor(input_image , flag),flag是转换类型：cv2.COLOR_BGR2GRAY,cv2.COLOR_BGR2HSV

HSV(Hue , Saturation , Value):色调，饱和度，明度
色度H:用角度度量，取值范围为0~360，红色开始按逆时针方向计算，红色为0度，绿色为120度，蓝色为240度
饱和度S:接近光谱色的程度，颜色可以看成是光谱色与白色混合结果，光谱色占的比例愈大，颜色接近光谱色的程度
        越高，颜色饱和度就越高。光谱色中白色成分为0，饱和度达到最高，取值范围0%~100%，值越大，颜色越饱和
明度V:表示颜色明亮的程度，对于光源色，明度值与发光体的光亮度有关；对于物体色，与物体的透射比有关，取值
      范围为0%(黑)~100%(白)
RGB面向硬件，HSV面向用户

在Opencv中
H色度取值范围是[0,179]
S饱和度的取值范围是[0,255]
V明度的取值范围是[0,255]
拿opencv的HSV值与其他软件的HSV值进行对比时，要归一化
'''

#获取颜色转换中所有可以使用的flag
def getColorConvertFlag():
    # dir() 查找module下的所有类
    flags = [i for i in dir(cv2) if i.startswith("COLOR_") ]
    print(flags)

'''''
物体跟踪，可以将图像从BGR转换到HSV后，提取某个特定颜色的物体
提取蓝色物体步骤：
1从视频中获取每一帧图像
2将图像转换到HSV空间
3设置HSV阈值到蓝色范围
4获取蓝色物体
'''
def trackObject():
    cap = cv2.VideoCapture(0)
    while(1):
        ret , frame = cap.read()

        #转换为hsv
        hsv = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV)
        #注意这里的上下限都是一个含有HSV的三元组
        lower_blue = np.array([110 , 50 , 50])
        upper_blue = np.array([130 , 255 , 255])
        '''''
        cv2.inRange(src , lowerb , upperb[,dst])
        作用：更改函数对某个单通道中的元素检查其值是否在范围中
        src:输入数组，lowerb:包含低边界的数组，upperb:包含高边界的数组，dst:输出数组
        如果src(I)符合范围，则dst(I)被设置为255，也就是说dst返回的是非黑即白的图像，而且符合要求
         的部分是白色的
        '''
        #构建物体掩膜（黑白部分），注意这里要使用hsv
        mask = cv2.inRange(hsv , lower_blue , upper_blue)
        #对原图像和掩膜进行位运算
        res = cv2.bitwise_and(frame ,frame , mask = mask)
        cv2.imshow("frame" , frame)
        cv2.imshow("mask" , mask)
        cv2.imshow("res" , res)
        k = cv2.waitKey(5) & 0xFF
        #ASCII中27是esc
        if k == 27:
            break
    cv2.destroyAllWindows()


'''''
如何找到要跟踪对象的HSV值,使用cv2.cvtColor,传入的参数是(你想要的)BGR值而不是一幅图。
例如找到绿色的HSV值，在终端输入以下命令
'''
def getHSV():
    '''''
    三层括号对应于:cvArray,cvMatIplImage
    也就是第一个括号是数组，第二个是矩阵，第三个是图像
    '''
    green = np.uint8( [ [ [0 , 255 , 0] ] ])
    hsv_green = cv2.cvtColor(green , cv2.COLOR_BGR2HSV)
    print(hsv_green)
    '''''
    可以分别用[H-100 , 100 , 100]和[H+100 , 255 , 255]做上下阈值，也可以用图像编辑软件(GIMP)
    '''

if __name__ == "__main__":
    getColorConvertFlag()
    '''
    ['COLOR_BAYER_BG2BGR', 'COLOR_BAYER_BG2BGRA', 'COLOR_BAYER_BG2BGR_EA',
     'COLOR_BAYER_BG2BGR_VNG', 'COLOR_BAYER_BG2GRAY', 'COLOR_BAYER_BG2RGB',
     'COLOR_BAYER_BG2RGBA', 'COLOR_BAYER_BG2RGB_EA', 'COLOR_BAYER_BG2RGB_VNG',
     'COLOR_BAYER_GB2BGR', 'COLOR_BAYER_GB2BGRA', 'COLOR_BAYER_GB2BGR_EA',
     ...
     'COLOR_YUV420sp2RGB', 'COLOR_YUV420sp2RGBA', 'COLOR_mRGBA2RGBA']
    '''
    # trackObject()  # 这个是连接摄像头的
    getHSV()
    '''
    [[[ 60 255 255]]]
    '''