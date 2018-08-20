import numpy as np
import cv2  # opencv

'''
python OpenCV 图像通道分离和合并 （四）
http://blog.csdn.net/sunny2038/article/details/9080047
图像通道分离有两种方法，一个是使用OpenCV自带的split函数；一个是使用Numpy数组来分离通道.
'''
# 使用OpenCV 自带 split函数
imagePath = '0-common_pics/common_1.jpg'
img = cv2.imread(imagePath)
b, g, r = cv2.split(img)
cv2.imshow("Blue", r)
cv2.imshow("Red", g)
cv2.imshow("Green", b)

'''
# 也可以单独返回其中一个通道 , 打印出图片的三种通道形式的图片。类似photoShop通道处理图片
'''
bb = cv2.split(img)[0]  # B通道
gg = cv2.split(img)[1]  # G通道
rr = cv2.split(img)[2]  # R通道

# 使用 Numpy 数组来实现图像通道分离
img = cv2.imread(imagePath)
print('img.shape ' ,img.shape)  # (540, 1000, 3) 行 列 ndim维度
print('img.ndim ' ,img.ndim)  # 3
# 创建3个跟图像一样大小的矩阵，数值全部为0
b = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
g = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
r = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)

#复制图像通道里的数据
b[:, :] = img[:, :, 0]  # 复制 b 通道的数据
g[:, :] = img[:, :, 1]  # 复制 g 通道的数据
r[:, :] = img[:, :, 2]  # 复制 r 通道的数据

cv2.imshow("Blue", b)
cv2.imshow("Red", r)
cv2.imshow("Green", g)

'''
通道合并也有两种方法。
1、使用OpenCV自带的merge函数 merged = cv2.merge([b,g,r]) #前面分离出来的三个通道
2、使用Numpy方法 mergedByNp = np.dstack([b,g,r])
'''
merged = cv2.merge([b, g, r])
print("Merge by OpenCV " ,merged.strides) # merge by OpenCV (3000, 3, 1)

mergedByNp = np.dstack([b, g, r])
print("Merge by NumPy ", mergedByNp.strides) # merge by Numpy (3000, 3, 1)
'''
numpy 中最新的版本已经删除了这个函数 np.stack，但是可以用其他函数代替：
stack()             Join a sequence of arrays along a new axis.
hstack()            Stack arrays in sequence horizontally (column wise).
dstack()            Stack arrays in sequence depth wise (along third dimension).
concatenate()       连结的；连锁的 Join a sequence of arrays along an existing axis.
vsplit ()           Split array into a list of multiple sub-arrays vertically.
np.stack((a,b),axis = 2) #三维 ---------np.dstack((a,b)) # 处理三维堆叠的
np.stack((a,b),axis = 1) #行   ---------np.hatack((a,b))
np.stack((a,b),axis = 0) #列   ---------np.vatack((a,b))
'''

# cv2.waitKey(0)等待无限长时间，待按esc关闭(或逐个鼠标点击关闭)所有窗口且完成运行。
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()