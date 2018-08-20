import cv2
import  numpy as np

'''
https://blog.csdn.net/haofan_/article/details/76618362

简述

生活中经常要用到各种要求的证件照电子版，红底，蓝底，白底等，大部分情况我们只有其中一种，
所以通过技术手段进行合成，用ps处理证件照，由于技术不到位，有瑕疵，
所以想用python&openCV通过代码的方式实现背景颜色替换，加强一下对于openCV的学习，锻炼一下编码水平。
'''
import cv2
import  numpy as np

img_path_person = r"0-common_pics/women.png"
img=cv2.imread(img_path_person)
#缩放
rows,cols,channels = img.shape
img=cv2.resize(img,None,fx=0.5,fy=0.5)
rows,cols,channels = img.shape
cv2.imshow('img',img)

'''
获取背景区域

首先将读取的图像默认BGR格式转换为HSV格式，然后通过inRange函数获取背景的mask。
HSV颜色范围参数可调节 https://blog.csdn.net/taily_duan/article/details/51506776
'''
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower_blue=np.array([78,43,46])
upper_blue=np.array([110,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
cv2.imshow('Mask', mask)

'''
蓝色的背景在图中用白色表示，白色区域就是要替换的部分，但是黑色区域内有白点干扰，所以进一步优化。

#腐蚀膨胀
'''
erode=cv2.erode(mask,None,iterations=1)
cv2.imshow('erode',erode)
dilate=cv2.dilate(erode,None,iterations=1)
cv2.imshow('dilate',dilate)

'''
处理后图像单独白色点消失。
替换背景色
遍历全部像素点，如果该颜色为dilate里面为白色（255）则说明该点所在背景区域，于是在原图img中进行颜色替换。

#遍历替换
'''
for i in range(rows):
    for j in range(cols):
        if dilate[i,j]==255:
            img[i,j]=(0,0,255)#此处替换颜色，为BGR通道
cv2.imshow('res',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
总结

最开始想直接通过遍历全图进行替换背景色，但是图像中难免有些像素点和背景色一样，
造成了干扰，导致最后结果不尽人意，所以想通过这种方法进行处理。显然最后有明显的ps痕迹。
最后贴上完整代码，不足之处欢迎各位指正！
'''
