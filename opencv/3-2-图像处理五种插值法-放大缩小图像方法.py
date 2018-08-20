'''
https://blog.csdn.net/jningwei/article/details/78822026
'''
# coding=utf-8

import cv2
"""
INTER_NEAREST | 最近邻插值
INTER_LINEAR | 双线性插值（默认设置）
INTER_AREA |  使用像素区域关系进行重采样
INTER_CUBIC  | 4x4像素邻域的双三次插值
INTER_LANCZOS4 |  8x8像素邻域的Lanczos插值
"""

if __name__ == '__main__':
    img = cv2.imread(r"0-common_pics/girl.jpg")
    height, width = img.shape[:2]

    # 缩小图像
    size = (int(width*0.8), int(height*0.7))
    shrink_NEAREST = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    shrink_LINEAR = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    shrink_AREA = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    shrink_CUBIC = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    shrink_LANCZOS4 = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)

    # 放大图像
    fx = 1.2
    fy = 1.1
    enlarge_NEAREST = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    enlarge_LINEAR = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    enlarge_AREA = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    enlarge_CUBIC = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    enlarge_LANCZOS4 = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LANCZOS4)

    cv2.imshow("shrink_NEAREST.jpg", shrink_NEAREST)
    cv2.imshow("shrink_LINEAR.jpg", shrink_LINEAR)
    cv2.imshow("shrink_AREA.jpg", shrink_AREA)
    cv2.imshow("shrink_CUBIC.jpg", shrink_CUBIC)
    cv2.imshow("shrink_LANCZOS4.jpg", shrink_LANCZOS4)

    cv2.imshow("enlarge_NEAREST.jpg", enlarge_NEAREST)
    cv2.imshow("enlarge_LINEAR.jpg", enlarge_LINEAR)
    cv2.imshow("enlarge_AREA.jpg", enlarge_AREA)
    cv2.imshow("enlarge_CUBIC.jpg", enlarge_CUBIC)
    cv2.imshow("enlarge_LANCZOS4.jpg", enlarge_LANCZOS4)

    if cv2.waitKey(0)==27:
        cv2.destoryAllWindows()