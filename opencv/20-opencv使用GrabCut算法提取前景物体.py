import cv2  # 安装ipencv
import numpy as np
from matplotlib import pyplot as plt

'''
opencv-使用 GrabCut 算法进行交互式前景提取
参考：
1、http://docs.opencv.org/3.3.0/  官方文档api
2、http://docs.opencv.org/3.3.0/d6/d00/tutorial_py_root.html 官方英文教程
3、https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
4、https://github.com/makelove/OpenCV-Python-Tutorial# 进阶教程
5、https://docs.opencv.org/3.3.0/index.html  官方英文教程
6、https://github.com/abidrahmank/OpenCV2-Python-Tutorials
7、https://www.learnopencv.com/
8、http://answers.opencv.org/questions/  OpenCV论坛
9、https://github.com/opencv/opencv   官方github
10、https://github.com/abidrahmank/OpenCV2-Python-Tutorials
参考：https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,iterCount,cv2.GC_INIT_WITH_RECT)
img - 输入图像
mask掩码 - 它是一个掩模图像，我们指定哪些区域是背景，前景或可能的背景/前景等。它由以下标志，
    cv2.GC_BGD，cv2.GC_FGD，cv2.GC_PR_BGD，cv2.GC_PR_FGD或简单地通过 0,1,2,3到图像。
rect - 以格式（x，y，w，h）包含前景对象的矩形坐标，
bdgModel后景色，fgdModel前景色 - 这些是内部由算法使用的数组。 您只需创建两个大小为（1,65）的np.float64类型零数组。
iterCount - 算法运行的迭代次数。
模式 - 它应该是cv2.GC_INIT_WITH_RECT或cv2.GC_INIT_WITH_MASK或组合，它决定我们是绘制矩形还是最终的触摸笔画。


首先看矩形模式。加载图像，创建一个类似的掩码图像。 创建fgdModel和bgdModel。
给出矩形参数。 这一切都是直截了当的。 让算法运行5次迭代。 模式应该是cv2.GC_INIT_WITH_RECT，
因为使用矩形。 然后运行grabcut。 它修改掩模图像。 在新的mask图像中，像素将被标记为四个标志，
表示上述的背景/前景。 所以修改掩码，使得所有的0像素和2像素都被置于0（即背景），所有的1像素和3像素被放到1（即前景像素）。
现在的最后面具准备好了。 只需将其与输入图像相乘即可获取分割图像。
'''
imagePath = '0-common_pics/colorCat.png'
img = cv2.imread(imagePath)
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (50,50,450,290)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img),plt.colorbar(),plt.show()

# newmask is the mask image I manually labelled
newmask = cv2.imread(imagePath,0)

# whereever it is marked white (sure foreground), change mask=1
# whereever it is marked black (sure background), change mask=0
mask[newmask == 0] = 0
mask[newmask == 255] = 1

mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()


'''
实际做的是，在油漆应用程序中打开输入图像，并向图像添加了另一个图层。
在油漆中使用画笔工具，在这个新图层上用黑色标记了白色和不需要的背景（如标志，地面等）的
未来前景（头发，鞋子，球等）。 然后用灰色填充剩余的背景。
然后在OpenCV中加载该掩码图像，编辑原始掩码图像，在新添加的掩码图像中得到相应的值。 检查以下代码：
'''