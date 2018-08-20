import cv2  # 安装ipencv
import aircv as ac

'''
Aircv 是一款基于Python-opencv2的目标定位，查找基于对象位置
其原理，就是根据像素BGR值的一个匹配过程。
# 直接安装即可 pip install aircv   # 其他安装 whl 文件及源码在百度云盘 搜索 aircv
https://www.helplib.com/GitHub/article_118432
https://www.oschina.net/p/aircv
'''

def draw_circle(img, pos, circle_radius, color, line_width):
    # print(img, pos, circle_radius, color, line_width)
    cv2.circle(img, pos, circle_radius, color, line_width)
    cv2.imshow('objDetect', imsrc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    imsrc = ac.imread('0-common_pics/bg.jpg')
    imobj = ac.imread('0-common_pics/obj.png')

    imsrc_, imobj_ = imsrc[:], imobj[:]

    pos = ac.find_template(imsrc_, imobj_) # find the match position
    # help( ac.find_template )
    # print( pos )  # {'result': (793.0, 450.0), 'rectangle': ((744, 405), (744, 495), (842, 405), (842, 495)), 'confidence': 0.543538510799408}

    circle_center_pos = pos['result']
    '''
    TypeError: integer argument expected, got float 参数预期应该是 整型，结果得到浮点型
    因为 circle_center_pos 是浮点型。所以要转整型为 new_int_circle_center_pos
    '''
    new_int_circle_center_pos = []
    for i in list(circle_center_pos):
        new_int_circle_center_pos.append(int(i))
    print( tuple(new_int_circle_center_pos) )  # (793, 450)

    circle_radius = 50
    color = (0, 255, 0)
    line_width = 10

    # draw circle
    draw_circle(imsrc_, tuple(new_int_circle_center_pos), circle_radius, color, line_width)

