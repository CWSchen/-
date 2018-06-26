'''
手写识别系统
构建识别类
Recognize
调用getResult()函数即可
'''

import operator
from numpy import *
from PIL import Image
from os import listdir
from io import BytesIO

def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]    #训练数据集的行数
    # 计算距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # 返还距离排序的索引
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), 
                              key=operator.itemgetter(1), reverse=True)
    
    return sortedClassCount[0][0]

# 将图片转化为行向量
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

'''
如何让加载训练集值运行一次？
'''
hwLabels , trainingMat = [] , []

def loadTrainingSet(dir_trainingSet):
    
    print('把trainingDigits文件夹里的所有训练集导入')
    #把trainingDigits文件夹里的所有训练集导入
    trainingFileList = listdir(dir_trainingSet)
    #print(trainingFileList)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024)) # 初始化训练矩阵
    for i in range(m):
        # 此三步，将所有训练集的名称分割只取出第一个
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        
        # 得到一个由训练集 名称首个number的矩阵
        hwLabels.append(classNumStr)
        
        # 每一个 训练集的 txt 都转成一个 1行1025列的向量
        trainingMat[i,:] = img2vector(dir_trainingSet+'/%s' % fileNameStr)

    return  hwLabels , trainingMat


def getResult(filename,trainingDigits):
    '''
    filename 测试集dir
    trainingDigits 训练集dir
    '''
    hwLabels , trainingMat = loadTrainingSet(trainingDigits)
        
    # 为输入的数字图片分类，读取图片为
    with open(filename, 'rb') as f:
        filePath = f.read()
    # 此时 filePath 是十六进制字节  如： \x7f\x12\xdf
    fileNameStr = changeImg2Text(filePath,filename)
    inputVect = img2vector(fileNameStr)
    
    classifierResult = classify(inputVect, trainingMat, hwLabels, 3)
    print( '预测手写数字识别为：',classifierResult)
    return classifierResult
    
    # 原demo里有这句话，可以这句话，会将预测的图片失效，暂注释 保留
    #with open(filename, 'w') as f:
    #    f.write(str(classifierResult))

# 处理初始图形
def changeImg2Text(filePath,filename):
    # 就是字符串 \ 分割后(其中 \\ 是加了转译)，取最后一个 2.jpg，再 以 . 分割取 名字
    fileNameStr = filename.split('\\')[-1].split('.')[0] + '.txt'
    fr = open(fileNameStr, 'w')
    
    #读图片转矩阵，Python 3 要加 BytesIO(filePath)
    '''
    https://codedump.io/share/aztOtkSsnO2U/1/python-valueerror-embedded-null-byte-when-reading-png-file-from-bash-pipe
    '''
    im = Image.open(BytesIO(filePath))
    #print(im) # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=206x376 at 0x8D99C50>
    im2 = im.resize((32, 32), Image.ANTIALIAS)
    img = array(im2)
    print( img.shape , Image.ANTIALIAS )
    
    m, n = img.shape[:2]

    for i in range(m):
        for j in range(n):
            R, G, B = img[i, j, :]
            # 因为，图片首先要 处理成灰度图，所以根据，灰度进而识别
            '''
            这部分的颜色用 PhotoShop 取色器，调参。
            RGB的值选择 白色点 和 目标颜色点的中点的RGB
            '''
            #if R < 40 and G < 40 and B < 40:   # 这些参数时对于黑白色的区分
            #if R < 245 and G < 153 and B < 120:   # 对 0 文件里，橙色图片的划分
            if R < 185 and G < 100 and B < 100:   # 对 2 文件里，灰色图片的划分
                fr.write('1')
            else:
                fr.write('0')
        fr.write('\n')

    fr.close()
    return fileNameStr
