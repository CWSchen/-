import cv2
import numpy as np
'''
多通道图像遍历
来源：https://blog.csdn.net/maryhuan/article/details/51330413


阅读数：928
方法一：普通遍历

for(int y = 0;y < height; y++)//行
    {
        for(int x = 0; x < width; x++)//列
        {
            Scalar pix = srcImg.at<Vec3b>(y,x);
            int channle0 = pix[0];
            int channle1 = pix[1];
            int channle2 = pix[2];
        }
    }

方法二：行指针，比较高效
<pre name="code" class="cpp">for(int y = 0;y < height; y++)//行
{
    Vec3b *pSrcData = srcImg.ptr<Vec3b>(y);//每一行的指针
    for(int x = 0; x < width; x++)//列
    {
        int channle0 = pSrcData[x][0];
        int channle1 = pSrcData[x][1];
        int channle2 = pSrcData[x][2];
    }
}

方法三：行指针，比方法二高效

for(int y = 0;y < height; y++)//行
    {
        uchar *pSrcData = srcImg.ptr<uchar>(y);//每一行的指针
        for(int x = 0; x < width; x++)//列
        {
            int channle0 = pSrcData[x*channel];
            int channle1 = pSrcData[x*channel+1];
            int channle2 = pSrcData[x*channel+2];
        }
    }

在每一行数据元素之间在内存里是连续存储的，因为图像在OpenCV里的存储机制问题，行与行之间可能有空白单元。
方法四：数据指针。高效

uchar *pSrcData = srcImg.data;//每一行的指针
    for(int y = 0;y < height; y++)//行
    {
        for(int x = 0; x < width; x++)//列
        {
            int channle0 = pSrcData[y*width*channel+x*channel];
            int channle1 = pSrcData[y*width*channel+x*channel+1];
            int channle2 = pSrcData[y*width*channel+x*channel+2];
        }
    }
一般来说图像行与行之间往往存储是不连续的，但是有些图像可以是连续的，Mat提供了一个检测图像是否连续的函数isContinuous()。当图像连通时，我们就可以把图像完全展开，看成是一行。


经过测试，时间分别是：

方法一:296

方法二:47

方法三:0

方法四:31
'''