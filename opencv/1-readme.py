import numpy as np
import cv2  # opencv
'''
二、图像基本知识
1、图像是什么：
    图像是客观对象的一种相似性的、生动性的描述或写真，是人类社会活动中最常用的信息载体。或者说图像是客观对象的一种表示，它包含了被描述对象的有关信息。
2、图像基本属性有哪些：
    通道数目、高与宽、像素数据、图像类型
'''
'''
opencv 等pythonlibs 下载网址
https://www.lfd.uci.edu/~gohlke/pythonlibs/

下载完成 opencv_python-3.4.1+contrib-cp35-cp35m-win_amd64.whl
将它放在与pycharm terminal 命令行的同一个文件夹里，如：此处是放置在 0-python 文件夹里，
然后 pip install opencv_python-3.4.1+contrib-cp35-cp35m-win_amd64.whl 即可安装
'''

'''
* [基本的Numpy教程](http://wiki.scipy.org/Tentative_NumPy_Tutorial)
* [Numpy示例列表](http://wiki.scipy.org/Numpy_Example_List)
* [OpenCV文档](http://docs.opencv.org/)
* [OpenCV论坛](http://answers.opencv.org/questions/)

opencv 中图像处理的一般流程——面向对象
https://blog.csdn.net/libin88211/article/details/20860983
图像处理算法工程师面试题
https://blog.csdn.net/ali_dongdong/article/details/74518607

OpenCV常见的优化问题和技巧
https://blog.csdn.net/guyuealian/article/details/78540206



##################

学习这个人的博客，到这各地址的文章
https://blog.csdn.net/maweifei/article/details/53932782
https://blog.csdn.net/wc781708249/article/details/78320644
继续向后翻看


opencv 官方文档
http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html

http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/tutorials.html
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#b-rotated-rectangle.

http://www.ruanyifeng.com/blog/2016/07/edge-recognition.html

https://www.cnblogs.com/mymickeyyang1221/p/8141717.html

http://www.guanggua.com/question/11337499-How-to-convert-an-image-from-npuint16-to-npuint8.html

https://github.com/188080501/JQTools

https://blog.csdn.net/u014365862/article/details/52652273

我怎么感觉 photoShop 就是调用的opencv 做的图像处理

CVPR 是IEEE Conference on Computer Vision and Pattern Recognition的缩写，
即IEEE 国际计算机视觉与模式识别会议 。该会议是由IEEE举办的计算机视觉和模式识别领域的顶级会议。- 详情百度百科
'''

'''
CVPR2015一些文章整理
2016年12月13日 22:29:37528人阅读 评论(0) 收藏  举报
 分类： CNN--ANN--Deep Learning（52）   DL_ML_CNN原理（16）
简单看了一部分CVPR2015的文章。整理了一下。其中我决定把精彩的文章加粗。主要是觉得有些文章只读了一遍，没有发现很多很有道理的point（尽管我承认他们的工作都花了很大的功夫，但是没有激起太大的兴趣去follow。也许有机会读第二遍的时候会再highlight）。另外MIT的博士生Zoya Bylinskii也总结了一个list，大家可以看看这里：http://web.mit.edu/zoya/www/CVPR2015brief.pdf
如果有不同看法的我们可以在评论区里讨论。

CNN结构的：
   ---  Fisher Vectors Meet Neural Networks: A Hybrid Classification Architecture，Florent Perronnin and Diane Larlus
         相比于标准的CNN，变化是将卷积层全部变成标准的FV，全连接层的部分做分类层保持不变。比起标准的FV，无疑是把分类器变成了MLP。ACC相比标准的CNN下降，相比标准的FV提高。这种从标准CNN入手，把前面的卷积和后面的全连通隔裂开对待/优化的文章还有arxiv上He Kaiming 的 Object Detection Networks on Convolutional Feature Maps。
    ---- Recurrent Convolutional Neural Network for Object Recognition
          Weichen师兄在讨论班上的推荐。把层次空间想象成序列空间，套上RNN，目的是为了使同一层的节点相互联系从而建模context。这个想法挺有脑洞，但是感觉很不自然（为什么不直接建模相邻节点的依赖关系）。相比之下ION net建模context的方法更直接，以后有机会会讲讲ION。

物体检测与分割：
   ---- Learning to Propose Object， Philipp Krähenbühl, Vladlen Koltun
   ---- Improving Object Proposals with Multi-Thresholding Straddling Expansion， Xiaozhi Chen, Huimin Ma, Xiang Wang, Zhichen Zhao
   ---- Hypercolumns for Object Segmentation and Fine-Grained Localization， Bharath Hariharan, Pablo Arbeláez, Ross Girshick, Jitendra Malik
        这个比较有意思了，明白说CNN每一层都是有用处的。Holistically-Nested Edge Detection的模型跟这个模型有相似的味道。
   ---- Taking a Deeper Look at Pedestrians
         这文章在方法上有啥创新点？好像就是把Cifar-net和Alexnet用在对行人的建模上。
   ---- A Convolutional Neural Network Cascade for Face Detection，Haoxiang Li，Gang Hua
        CNN + Cascade，Calibration层有点意思，模型里还引入了multi-scale。
   ---- Deeply learned face representations are sparse, selective, and robust, Yi Sun, Xiaogang Wang, Xiaoou Tang
        DeepID系列之DeepID2+。在DeepID2之上的改进是增加了网络的规模(feature map数目)，另外每一层都接入一个全连通层加supervision。最精彩的地方应该是后面对神经元性能的分析，发现了三个特点：1.中度稀疏最大化了区分性，并适合二值化；2.身份和attribute选择性；3.对遮挡的鲁棒性。这三个特点在模型训练时都没有显示或隐含地强加了约束，都是CNN自己学的。已经迫不及待要看DeepID3了。
   ---- DeepID3: Face Recognition with Very Deep Neural Networks （顺带提一下吧）
        DeepID3似乎是封山之作，结论是太Deep了在现有数据集上也没什么提升了。反正作者也毕业了。CSDN有一篇对作者的专访，见：http://www.csdn.net/article/2015-11-18/2826241
   ---- Hypercolumns for Object Segmentation and Fine-Grained Localization， Bharath Hariharan, Pablo Arbeláez, Ross Girshick, Jitendra Malik
        这个比较有意思了，明白说CNN每一层都是有用处的。Holistically-Nested Edge Detection的模型跟这个模型有相似的味道。
   ---- Fully Convolutional Networks for Semantic Segmentation (Best Paper Honorable Mention), Jonathan Long, Evan Shelhamer, Trevor Darrell
        文章把全连接层当做卷积层，也用来输出feature map。这样相比于Hypercolumns/HED 这样的模型，可迁移的模型层数（指VGG16/Alexnet等）就更多了。但是从文章来看，因为纯卷积嘛，所以feature map的每个点之间没有位置信息的区分。相较于Hypercolumns的claim，鼻子的点出现在图像的上半部分可以划分为pedestrian类的像素，但是如果出现在下方就应该划分为背景。所以位置信息应该是挺重要需要考虑的。这也许是速度与性能的trade-off?
   ----- Is object localization for free - Weakly-supervised learning with convolutional neural networks
       弱监督做object detection的文章。首先fc layer当做conv layer与上面这篇文章思想一致。同时把最后max pooling之前的feature map看做包含class localization的信息，只不过从第五章“Does adding object-level supervision help classification”的结果看，效果虽好，但是这一物理解释可能不够完善。

（PS. arxiv上有三篇借助CNN做一般物体检测的：
   ---- DeepBox: Learning Objectness with Convolutional Networks，Weicheng Kuo，Bharath Hariharan，Jitendra Malik
        没太大意思，就是把CNN用在所有物体类的训练上。另外证明学到的模型是generic的时候用了IOU-0.5的准确率而不是0.8或者AR是没有很高信服度的。（ICCV2015接收）
   ---- Boosting Convolutional Features for Robust Object Proposals, Nikolaos Karianakis
        把VGG第一层输出当做feature channel然后接boosting做分类。并没有证明算法的一般性。
   ---- Learning to Segment Object Candidates， Pedro O. Pinheiro， Ronan Collobert， Piotr Dollar （NIPS2015接收）
        文章好像没讲明白score那个分支训练集是如何做出标注的（@8.7又读了一遍，如何标注就靠正样本选取时的constraints，自己第一遍的时候没弄明白）。segment相比bounding box在速度上也有点吃亏，所以5秒一个图算慢的（其实5秒就能过一个图还是很快的啊，用的是VGG16的网络）。但比起MCG这速度还是快多了。
        另外Microsoft COCO今年被用起来了。Microsoft COCO也做成竞赛了，好像Detection Task今年在ICCV15要和ILSVR合办workshop。)

CNN做边缘轮廓检测：
   ---- DeepContour： A Deep Convolutional Feature Learned by Positive-sharing Loss for Contour Detection
         二分类变多分类，有点joint learning的意思。
   ---- DeepEdge A Multi-Scale Bifurcated Deep Network for Top-Down Contour Detection
         相当于一种multi-clues做二分类问题。文章里的multi-scale和上面CNN+Cascade那篇文章模型里用到的multi-scale不是同一个东西，用DSP-SIFT一文的总结就是，本文说的multi-scale只是在size-space中选了多个size，并不是CNN+Cascade一文中在scale-space中选择了多个scale。multi-scale是解决真正的不同尺度的多样性，而multi-size更像是引入不同的context以及克服occlusion。个人理解这两点的目标区别于此。

PS. 上面两篇相比传统方法提高并不明显。看来在比较底层的问题上人工特征与end-to-end学习模型相比没有在high-level计算机视觉任务上差距的大。
arxiv上Tu Zhuowen有一篇性能更高的，优势还是很明显的（因为逐像素检测相比全图检测，失去了全局信息。这也隐含了R-CNN的缺点吧）：
   ---- Holistically-Nested Edge Detection
        分析了各种multi-scale model，Wang Naiyan在VALSE的tutorial上也用了这个论文的插图。这个模型很复杂了，除了讨论multi-scale以外，还叠加了cnn multi-layer的区分性，有点Hypercolumns的味道。（ICCV2015接收）


利用CNN的局部性解决计算机视觉问题：
   ---- A Discriminative CNN Video Representation for Event Detection，Zhongwen Xu, Yi Yang, Alex G. Hauptmann
        CNN conv5输出可以作为concept detector。valse上的ppt：这里。
   ---- Exploiting Local Features from Deep Networks for Image Retrieval
        Workshop paper，与上文的思路如出一辙，不过证明了在检索过程中concept概念越抽象不一定越好--因为搜索毕竟是instance-level的，不是class-level的。

图像检索的：
   ---- Query-Adaptive Late Fusion for Image Search and Person Re-Identification
         郑博每年都有CVPR，恭喜。在valse上的ppt：这里。我们在Trecvid2015的竞赛中用了这个方法，很多人当时也觉得这项工作很有意义。
   ---- Early Burst Detection for Memory-Efficient Image Retrieval， Miaojing Shi, Yannis Avrithis, Hervé Jégou
         Hervé Jégou也加入FAIR了
   ---- Pairwise Geometric Matching for Large-scale Object Retrieval
         利用Geometry information做 verification的。速度还挺快。

Eye-fixation:
   ----  Predicting Eye Fixations Using Convolutional Neural Networks, Nian Liu, Junwei Han, Dingwen Zhang, Shifeng Wen, Tianming Liu
         之前没太关注eye-tracking data。这篇文章就是用预测eye fixation的，跟显著性有比较大的联系。这篇文章中利用的multi-resolution的模型，在看过其他文章之后不会觉得有特别特殊的地方，但是从一个contrast导致saliency的角度去结束这里用到的multi-resolution模型，还有点意思。（add@Nov/09/2015: 其实在Naiyan Wang在VALSE上的总结，Saliency和Edge Detection、Segmentation类似，都是做pixel-wise labeling，所以这几个问题都是同质的，所以用相似的模型去解决完全合理。）
   ----  Eye Tracking Assisted Extraction of Attentionally Important Objects From Videos， Karthikeyan Shanmuga Vadivel, Thuyen Ngo, Miguel Eckstein, B.S. Manjunath
        Manj组今年唯一的CVPR论文了，用eye-tracking数据辅助其他（指除了saliency）computer vision task，这里做的是video里的objectness。
   ---- Salient Object Subitizing
        数图像中显著物体的个数。好处是有的图像没有显著物体，而一般的Salient Object Detection方法仍然会检测出几个object。所以事前估计图像显著物体的数目可以作为一个有效的先验（比如没有显著物体的图像就不做检测了）。模型放在caffe的model zoo里了。
   ---- SALICON: Saliency in Context
        一个新库，拿MsCOCO标注的。理由是eye-tracking data的采集需要专门设备，不便于众包，所以她们组用鼠标轨迹代替eye-tracking data采集了human gaze的数据，而且证明了这种采集方法替代eye-tracking很合理。并且她们开放了一个新的竞赛就叫SALICON。还有后续的论文在ICCV2015上，以后专门讲ICCV15的论文时候再说。
   附arxiv上近期放出的论文：
   ---- DeepSaliency：Multi-task deep neural network model for salient object detection
         这里的multi-task是指semantic segmentation + salient object segmentation。不同于joint learning（如DeepID2和Fast RCNN），这里的两个task只是共享了conv layers，输入的训练样本是不一样的。训练的时候两个任务迭代地更新网络的参数。
   ---- DeepFix：A Fully Convolutional Neural Network for predicting Human Eye Fixations
        在MIT的saliency库上排在第二名。很有意思的文章，考虑了Fixation Prediction的Center Bias问题（就是人眼显著性判决时会倾向于图像中心。FCN这类模型因为没有全连接层了，所以输出每个像素的预测值是与位置无关的）。至于怎么解决的，请大家自行去看。

其他不好分类：
   ---- MatchNet Unifying Feature and Metric Learning for Patch-Based Matching，  Xufeng Han， Thomas Leung， Yangqing Jia， Rahul Sukthankar，Alexander C. Berg
         wide-baseline matching，相比与arxiv14年的Descriptor Matching with Convolutional Neural Networks a Comparison to SIFT，这篇文章是监督的，上篇文章是无监督的。patch matching其实和face verification、再辨识的关联挺大的。文中有说到测试的时候采用两步测试的方法：第一步是特征提取（过一个Tower就行），第二步是matching（把两个Tower的特征比较起来），这样先把第一步做完，特征保存起来，做第二步就容易了。联想道Valse上王晓刚老师将NIPS14那篇Joint identification and verification一文，王老师说verification那个网络的时候提到的缺点，不就可以用这个两步测试的方法来解决吗？
   ---- Domain-Size Pooling in Local Descriptors: DSP-SIFT ， Jingming Dong，Stefano Soatto
         wide-baseline matching，相比前面的MatchNet，这篇文章是无监督的。这篇文章Figure8解释了scale-space和size-space的概念，解释的非常好。但是DoG为什么归为size-space？我仍然觉得DoG是属于scale-space的。
   ---- Deep Neural Networks are Easily Fooled    （深度学习对抗样本）
   ---- Age and Gender Classification using Convolutional Neural Networks
         CNN做性别和年龄判决的。年龄判决不是用回归，而是把年龄分组，然后用分类的方法做。有点简单。而且Age和Gender分了两个网络分别做，竟然没有联合起来做。


还在看，慢慢整理吧。

另外这里有其他大神做的CVPR2015年的整理和总结：
CVPR 2015 之深度学习篇(3贴)：
   http://deepnn.net/viewtopic.PHP?f=6&t=31
   http://deepnn.net/viewtopic.php?f=6&t=32
   http://deepnn.net/viewtopic.php?f=6&t=38
武汉大学张觅博士生（原创）：CVPR 2015会议总结报告：
   http://valseonline.org/thread-334-1-1.html
(知乎)CVPR 2015 有什么值得关注的亮点？
   http://www.zhihu.com/question/31300014
Deep down the rabbit hole: CVPR 2015 and beyond:
   http://www.computervisionblog.com/2015/06/deep-down-rabbit-hole-cvpr-2015-and.html

-------
jiang1st
http://jiangwh.weebly.com
'''

'''
 计算机视觉的三大会议
2016年12月13日 22:40:21497人阅读 评论(0) 收藏  举报

1 ICCV 全称 IEEE International Conference on Computer Vision，国际计算机视觉大会

2 ECCV 全称 Europeon Conference on Computer Vision

3 CVRP 国际计算机视觉与模式识别学术会议
    即 International Conference on Computer VisionPattern Recognition
'''
