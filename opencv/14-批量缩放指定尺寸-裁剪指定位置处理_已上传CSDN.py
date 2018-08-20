import cv2
import os , sys
# print(sys.path)
'''
做图像处理需要大批量的修改图片尺寸来做训练样本，为此本程序借助opencv来实现大批量的剪切图片。
类似 photoShop 里的动作，批处理文件夹里的图片
'''

def cutimage(dir, suffix):
    for root, dirs, files in os.walk(dir):
        print(root, dirs, files)
        for file in files:
            filepath = os.path.join(root, file)  # 拼合路径
            filesuffix = os.path.splitext(filepath)[1][1:] # 截取文件后缀名/扩展名
            if filesuffix in suffix:  # 遍历找到指定后缀的文件名["jpg",png]等
                image = cv2.imread(filepath)  # opencv剪切图片，得到三维矩阵数据
                # cv2.imshow(file,image)

                # 缩放至定尺寸w*h
                dim = (242, 200)
                '''这里采用的插值法是INTER_LINEAR  '''
                resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                cv2.imshow("resize:%s" % filepath, resized)  #显示图像
                cv2.imwrite("0-common_pics/bank-icon-batch/%s" % file, resized)  #保存文件

                '''
                #获取截取区域
                cropImg = image[100:200,350:500]
                cv2.imshow("resize:%s" % filepath, cropImg) #显示图像
                cv2.imwrite("0-common_pics/bank-icon-cut/%s" % file, cropImg)  #保存文件
                '''


suffix = ['jpg', 'png']
dir = '0-common_pics/bank-icon'  # 文件夹要批量处理图片的原图片路径
cutimage(dir, suffix)

k = cv2.waitKey(0)
if k==27: # 按ESC
    cv2.destroyAllWindows()
