import cv2 as cv
import numpy as np

'''
https://blog.csdn.net/gangzhucoll/article/details/79008137
'''
def watershed(img):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(mb, kernel, iterations=3)
    dist = cv.distanceTransform(mb, cv.DIST_L2, 3)
    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)
    ret, surface = cv.threshold(dist, dist.max() * 0.6, 255, cv.THRESH_BINARY)
    surface_fg = np.uint8(surface)
    unknown = cv.subtract(sure_bg, surface_fg)
    ref, markers = cv.connectedComponents(sure_bg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(src, markers=markers)
    src[markers == -1] = [0, 0, 255]
    cv.imshow("result",src)


common_pics_path = "0-common_pics/common_1.jpg"
src = cv.imread(common_pics_path)
cv.imshow('def', src)

watershed(src)
cv.waitKey(0)
cv.destroyAllWindows()