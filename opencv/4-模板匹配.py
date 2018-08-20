'''
Python3与OpenCV3.3 图像处理(十四)--模板匹配
2017年12月07日 23:39:30
阅读数：750
https://blog.csdn.net/gangzhucoll/article/details/78747256

一、什么是模板匹配

在整个图像区域发现与给定子图像匹配的区域，模板匹配的工作方式是在待检测图像上从左到右，从上到下计算模板图象与重叠子图像的匹配度，匹配度越大，两者越相同

二、OpenCV中的模板匹配

CV_TM_SQDIFF 平方差匹配法：该方法采用平方差来进行匹配；最好的匹配值为0；匹配越差，匹配值越大。
CV_TM_CCORR 相关匹配法：该方法采用乘法操作；数值越大表明匹配程度越好。
CV_TM_CCOEFF 相关系数匹配法：1表示完美的匹配；-1表示最差的匹配。
CV_TM_SQDIFF_NORMED 归一化平方差匹配法
CV_TM_CCORR_NORMED 归一化相关匹配法
CV_TM_CCOEFF_NORMED 归一化相关系数匹配法
'''
import cv2 as cv
import numpy as np


butterfly = "0-common_pics/butterfly.jpg"
butterfly_bg = "0-common_pics/butterfly_bg.jpg"
def template():
    '''#模板图片 / 要匹配的小图片'''
    tpl=cv.imread(butterfly)
    '''#目标图片 / 被匹配的大图片'''
    target=cv.imread(butterfly_bg)
    cv.imshow('template',tpl)
    cv.imshow('target',target)

    methods=[cv.TM_SQDIFF_NORMED,cv.TM_CCORR_NORMED,cv.TM_CCOEFF_NORMED]

    #获得模板的高宽
    th,tw=tpl.shape[:2]
    for md in methods:

        #执行模板匹配
        #target：目标图片
        #tpl：模板图片
        #匹配模式
        result=cv.matchTemplate(target,tpl,md)
        #寻找矩阵(一维数组当作向量,用Mat定义) 中最小值和最大值的位置
        min_val,max_val,min_loc,max_loc=cv.minMaxLoc(result)
        if md==cv.TM_SQDIFF_NORMED:
            tl=min_loc
        else:
            tl=max_loc

        br=(tl[0]+tw,tl[1]+th)
        #绘制矩形边框，将匹配区域标注出来
        #target：目标图像
        #tl：矩形定点
        #br：举行的宽高
        #(0,0,255)：矩形边框颜色
        #2：矩形边框大小
        cv.rectangle(target,tl,br,(0,0,255),2)
        cv.imshow('match-'+np.str(md),target)


template()

cv.waitKey(0)
cv.destroyAllWindows()