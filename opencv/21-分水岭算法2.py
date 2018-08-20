import cv2 as cv
import numpy as np

'''
https://blog.csdn.net/qq_30490125/article/details/53050058 - 已打印
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np


#我们在下面自行创建了原始图片
img = np.zeros((400, 400), np.uint8)
cv2.circle(img, (150, 150), 100, 255, -1)
cv2.circle(img, (250, 250), 100, 255, -1)

#进行了距离变换
dist = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_3)#euclidean distance
'''
这里先插入一下opencv的distance type：  前面有CV_ 的都是老版本的写法，新版本无 CV_
CV_DIST_USER =-1, /* User defined distance */
CV_DIST_L1 =1, /* distance = |x1-x2| + |y1-y2| */
CV_DIST_L2 =2, /* the simple euclidean distance */
CV_DIST_C =3, /* distance = max(|x1-x2|,|y1-y2|) */
CV_DIST_L12 =4, /* L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1)) */
CV_DIST_FAIR =5, /* distance = c^2(|x|/c-log(1+|x|/c)), c = 1.3998 */
CV_DIST_WELSCH =6, /* distance = c^2/2(1-exp(-(x/c)^2)), c = 2.9846 */
CV_DIST_HUBER =7 /* distance = |x|
'''

#单通道向三通道转换，watershed只接受三通道的图像
dist3 = np.zeros((dist.shape[0], dist.shape[1], 3), dtype = np.uint8)
dist3[:, :, 0] = dist
dist3[:, :, 1] = dist
dist3[:, :, 2] = dist

#创建分类的图层，包含种子点
markers = np.zeros(img.shape, np.int32)
cv2.circle(markers, (150, 150), 100, 0, -1)
cv2.circle(markers, (250, 250), 100, 0, -1)
markers[150,150] = 1 # seed for circle one
markers[250, 250] = 2 # seed for circle two
markers[50, 50] = 3 # seed for background
markers[350, 350] = 4 # seed for background

#执行分水岭算法
cv2.watershed(dist3, markers)
plt.imshow(markers)
plt.show()
'''
但如果我们仔细看过上面的原理分析，就会发现分水岭内部是通过判断梯度来决定下一个种子点的选择的。
在那张距离变换的图上，背景部分的梯度永远都是0，所以背景部分会优先与山峰部分被标记，
知道背景部分遇到了山峰的边缘部分，这个时候的梯度跟山顶部分的梯度相当，
这时，山顶的种子才有可能后的向四临域泛洪的机会，
所以两个种子的标签类最终会在山腰部分相遇，形成分水岭，这个山腰大概是１／２圆半径的位置。
'''