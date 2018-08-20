import cv2
import  numpy as np

'''
https://blog.csdn.net/haofan_/article/details/76687238
获取目标区域（抠图）将目标区域和背景分离开。
此方法抠图只适合颜色对比比较明显的图片，允许存在少量颜色干扰
加载&缩放
通过imread函数加载图片，resize函数对图像进行缩放。
'''
img_path_person = r"0-common_pics/women.png"
img_path_bg = r"0-common_pics/common_1.jpg"
img=cv2.imread(img_path_person)
img_back=cv2.imread(img_path_bg)

#缩放
rows,cols,channels = img_back.shape
img_back=cv2.resize(img_back,None,fx=0.7,fy=0.7)
cv2.imshow('img_back',img_back)


rows,cols,channels = img.shape
img=cv2.resize(img,None,fx=0.4,fy=0.4)
cv2.imshow('img',img)
rows,cols,channels = img.shape #rows，cols最后一定要是前景图片的，后面遍历图片需要用到

'''
要实现的效果就是，把人物图像抠出来，放在背景图片上面。

获取背景区域

由于背景纯蓝色，所以找到了这些区域，相反的就是我们想要的。
这里要用到inRange这个函数获取蓝色区域。
首先需要将图片转换为HSV类型。
'''
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

'''
获取mask得到蓝色区域
'''
lower_blue=np.array([78,43,46])
upper_blue=np.array([110,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
cv2.imshow('Mask', mask)

'''
得到黑白型，白色为原蓝色区域。黑色是任务区域。

黑色区域有明显白点，有少量的颜色干扰，需要进一步优化。

mask优化

通过腐蚀和膨胀操作进行消除个别白点。
我对于腐蚀和膨胀操作的理解是：

腐蚀操作将会腐蚀图像中白色像素，以此来消除小斑点，
而膨胀操作将使剩余的白色像素扩张并重新增长回去。
#腐蚀膨胀
'''
erode=cv2.erode(mask,None,iterations=1)
cv2.imshow('erode',erode)
dilate=cv2.dilate(erode,None,iterations=1)
cv2.imshow('dilate',dilate)

'''
黑色区域内白点已经消除，完美分离人物与背景[傲娇]。

替换背景图片

此时已经将图片目标区域抠出来了，只需要再新的背景图上把抠出来的对应点颜色填充上去就好。
我们首先要确定一个坐标点，这个点决定了要把抠出来的图像放到新背景图片的什么位置，即就是抠出图片左上角（0，0）点在新的背景图片中应该在的位置。
注意：

扣出的图片应该小于背景图片，确定位置时候应注意，坐标越界后
会发生异常。注意协调。
#遍历替换
'''
center=[50,50]#在新背景图片中的位置
for i in range(rows):
    for j in range(cols):
        if dilate[i,j]==0:#0代表黑色的点
            img_back[center[0]+i,center[1]+j]=img[i,j]#此处替换颜色，为BGR通道
cv2.imshow('res',img_back)

cv2.waitKey(0)
cv2.destroyAllWindows()