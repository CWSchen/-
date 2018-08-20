'''
https://blog.csdn.net/gangzhucoll/article/details/78726069
'''
import cv2 as cv
import numpy as np


def create_rgb_hist(image):
    """创建rgb 三通道直方图"""
    h,w,c=image.shape
    rgbHist=np.zeros([16*16*16,1],np.float32)
    bsize=256/16
    for row in range(h):
        for col in range(w):
            b = image[row,col,0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index=np.int(b/bsize)*16*16+np.int(g/bsize)*16+np.int(r/bsize)
            rgbHist[np.int(index),0]=rgbHist[np.int(index),0]+1
    return rgbHist

def hist_compare(image1,image2):
    hist1=create_rgb_hist(image1)
    hist2=create_rgb_hist(image2)
    #巴氏距离比较，距离越小越相似
    match1=cv.compareHist(hist1,hist2,cv.HISTCMP_BHATTACHARYYA)
    #相关性比较，相关性越大越相似
    match2=cv.compareHist(hist1,hist2,cv.HISTCMP_CORREL)
    # 卡方比较，越大越不相似
    match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
    print("巴氏距离：%s 相关性：%s 卡方：%s"%(match1,match2,match3))


pic_1 = '0-common_pics/hist_1.png'
''' 这两张图片是直方图处理后保存的 '''
pic_clahe = '0-common_pics/hist_clahe.png'
pic_equalHist = '0-common_pics/pic_equalHist.png'

hist_compare(cv.imread(pic_1),cv.imread(pic_equalHist))
hist_compare(cv.imread(pic_1),cv.imread(pic_equalHist))