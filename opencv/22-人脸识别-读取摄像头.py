import cv2 as cv
import numpy as np

'''
《OpenCV3计算机视觉Python语言实现》此书已有
  人脸识别笔记
https://blog.csdn.net/rencia/article/details/79779365

老衲最近在学习《OpenCV3计算机视觉Python语言实现》中文版，
学到第五章人脸识别时懵逼了，书上代码都是片段，不会搞啊，不死心找了英文版官网的源代码，
找了个遍，发现原版源代码排版错误，把第四章的代码贴到了第五章里。
网上搜了一大圈，貌似各位大神都没怎么说到这章。肿么办，硬办。
老衲来死磕，下面给出我的方法。老衲平板电脑是七彩虹的i818w，CPU是Z3735F，内存2G，32位操作系统。
装了Anaconda2，opencv3.0.0，python是2.7。

第一步，生成人脸识别数据。还好这段代码比较全，也是正确的。
'''
import cv2


def generate():
    # 老衲把脸的特征文件放在了C盘，用绝对路径调用，书上是相对路径，xml文件在opencv/sources/data/haarcascades里，拷到下面的路径里
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # 打开摄像头，老衲用的是7寸windows平板，带两个摄像头，1是前置，0是后置
    camera = cv2.VideoCapture(1)
    count = 0
    while (True):
        ret, frame = camera.read()
        # 作为初学者，老衲怀疑所有的新东西，print一下看看ret的输出
        print(ret)

        # 把摄像头的每一帧图像转换成灰度图像，这时书上就比较乱了
        # 有用cv2.cvtColor(frame, 1)也有用下面的，其实都一样
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 检测人脸，没好多说的，自己网上查大神们写的吧，不解释
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # 接下来就是循环保存图片了
        for (x, y, w, h) in faces:
            # 先画一个正方形，这很简单
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 接下来把图像全部格式化成200x200像素
            f = cv2.resize(gray[y: y + h, x:x + w], (200, 200))

            # 保存图片为opencv专用的*.pgm格式
            cv2.imwrite('E:\%s.pgm' % str(count), f)
            count += 1

        # 把咱们的老脸显示在camera名字的窗口里
        cv2.imshow("camera", frame)
        # 这里就个人理解是图像每秒12帧，当按下q键时退出while循环
        if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
            break
    # 释放摄像头句柄
    camera.release()
    # 销毁窗口，这句很重要，书上老漏导致老衲经常死机
    cv2.destroyAllWindows()

# 开始执行
# if __name__ == "__main__":
    # generate()

'''
 第二步，本着严谨的态度，老衲必须查看刚刚保存的人脸特征图，就是老衲的脸啦。用代码显示出来。
'''
# 打开C盘路径下保存图片的12.pgm文件，并保存为灰度图像
img = cv2.imread('E:\\MyW\\date\\12.pgm', cv2.IMREAD_GRAYSCALE)
# 顺便看看图片的格式，好大的一个列表对象，
# 里面的数组代表了图片上一个个行和列上的像素，格式是[xxx,xxx,xxx]
# xxx = 0~255
print(img)
# 在名为img的窗口上显示图片，像素为200x200
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()

'''
 第三步，上正菜，老衲觉得出版社太坑了，给了很多片段，
 还不给全部源代码，老衲一向自给自足丰衣足食，来看下面老衲整理的一大片代码。
'''
# coding=utf-8
import os
import sys
import cv2
import numpy as np

# 图片的路径
filepath = 'E:\\MyW\\date\\'
# 定义2个数组，X存放每幅图片的数组列表，y存放每幅图片的序号，后面有句print函数
# 可以在IDLE里看哪张图片特征最匹配实时检测到的脸，并给出置信度
X = []
y = []

# 顾名思义，读取特征图片
def read_images(path):
    # 初始化计数器
    c = 0

    # 扫描路径下的路径名，文件名，不明白的可以在下面print一下
    for dirname, dirnames, filenames in os.walk(path):
        # print dirname, dirnames, filename
        # 提取每个文件并保存到X,y数组里，这里老衲做了点改动，应为老衲的特征图片路径没有书上代码那么深
        for filename in filenames:
            try:
                # 组合路径和文件名，得到特征图的绝对路径c:\MyW\date\1.pgm
                filename = os.path.join(path, filename)
                # 把特征图以灰度图读取
                im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

                # 重新格式化图片为200x200像素，原书估计打错字了
                if (im is not None):
                    im = cv2.resize(im, (200, 200))

                # 把特征图片数组添加到X数组中，组成一个大的特征数组
                X.append(np.asarray(im, dtype=np.uint8))
                y.append(c)
            # 输入输出错误检查
            except (errno, strerror):
                print("I/O error({0}): {1}".format(errno, strerror))

            except:
                print("Unexpected error:", sys.exc_info()[0])
                raise
            c = c + 1
    #print X
    #print y
    # 估计到这，这数组的维度大得吓人了
    return [X, y]


# 顾名思义，人脸检测开始了
def face_rec():
    # 定义一个名字的数组，随便瞎打几个英文字母，等会下面会用到
    names = ['SHG', 'JSY', 'LYF']
    # 加载特征图片
    [x, y] = read_images(filepath)
    # 把y数组保存为int32格式的数组，用asarry()不用开辟新的内存，其实老衲觉得array()函数也可以，现在谁的电脑内存没个几G啊
    y = np.asarray(y, dtype=np.int32)
    # 加载EigenFaceRecognizer算法，这里必须改为EigenFaceRecognizer_create，原书createEigenFaceRecognizer
    # 因为老衲用的是opencv_contrib_python-3.4.0.12-cp27-cp27m-win32.whl
    # 如何安装contrib请各位施主自行百度，后面会讲一个老衲安装时碰到的小故事
    model = cv2.face.EigenFaceRecognizer_create()
    # 训练数据集，貌似机器学习，好高深，不深究
    model.train(np.asarray(X), np.asarray(y))

    # 和第一步里generate()里的用法一样，懒得解释了
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(1)

    while (True):
        read, img = camera.read()
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[x: x + w, y: y + h]
            try:
                # 选出感兴趣的区域，使用内插法，还是老规矩自行百度
                roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                # 预测实时图片，这里老衲也没看明白，下次有时间去看看EigenFaceRecognizer的方法
                params = model.predict(roi)
                # 把匹配的特征和置信度打印在IDLE内
                print("Label: %s, Confidence: %.2f" % (params[0], params[1]))
                # 把匹配的名字显示在方框左上角，有时候会瞎显示，以后研究，还有就是现在无法显示中文字符，也以后吧 :P
                cv2.putText(img, names[params[0]], (x, y - 20), \
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            except:
                continue
        cv2.imshow("camera", img)
        if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_rec()


