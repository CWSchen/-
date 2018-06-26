from handWriteRecognizeSystem import *
from os import listdir
import sys
'''
JPG是有损压缩，算法不一样出来结果是会有差别的。用BMP格式就不会有这个问题了。
'''

'''
训练图片 9.jpg 报错如下。
IndexError: too many indices for array   报这个错主要就是 矩阵的维度 没有对应上

是因为，在PhotoShop里查看此照片的图像模式，是灰度图，所以它的shape被读取为 二维的(32, 32)，
所以需要在PS里将图片用批量动作的方式将图片的图像模式转为 RGB模式。就可以读取为(32,32,3)
'''

#getResult('examplePictures\9_width_200.jpg','trainingDigits')
'''
强图片写为了 二进制的 ， 1的部分为手写体图像部分，0的部分指其他

模型训练问题总结：
这个模型，对于图片的width有要求
如果图片上手写体和边缘过近，会导致识别错误率增加

再者如果读写的矩阵是对的，但是模型不认识，就可以，把这个加到训练集里。让模型认识。
结果也不好使。训练集里的 9_204.txt 是我加入的。因为，9_width_200。jpg 得到的矩阵是对的，可是模型识别错了。


isinstance(i,str): 判断是否为 字符串
type(i)  输出类型
'''

#print(sys.path)

def run_result(dir,trainingDigits):
    testFiles = listdir('examplePictures/'+dir+'/')
    for i in testFiles:
        if i.split('.')[-1] in ['jpg','jpeg','gif','png','bmp' ]:
            print('-'*30)
            print('examplePictures/'+dir+'/'+i)
            getResult(r'examplePictures/'+dir+'/'+i,trainingDigits)

     
#run_result('0','trainingSet_0')  # 我后加的 电子数字 识别
'''
预测的不好的，倾向谁的，就把这个训练集删除
例如，把 3 预测成 9 了，就增加 3的训练集。
'''

#run_result('2','trainingSet_2')  # 后加的 电子数字图片 识别训练

run_result('1','trainingDigits')  # 项目原始的 手写体 数字识别
'''
examplePictures/2.jpg
把trainingDigits文件夹里的所有训练集导入
(32, 32, 3) 1
预测手写数字识别为： 2
examplePictures/3.jpg
把trainingDigits文件夹里的所有训练集导入
(32, 32, 3) 1
预测手写数字识别为： 3
examplePictures/9_width_200.jpg
把trainingDigits文件夹里的所有训练集导入
(32, 32, 3) 1
预测手写数字识别为： 7   # 这个识别错了，难道是因为 9 太棒了？被识别为 7了？
examplePictures/9_width_737.jpg
把trainingDigits文件夹里的所有训练集导入
(32, 32, 3) 1
预测手写数字识别为： 9
examplePictures/9_width_826.jpg
把trainingDigits文件夹里的所有训练集导入
(32, 32, 3) 1
预测手写数字识别为： 9
'''
