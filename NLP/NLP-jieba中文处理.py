# encoding=utf-8
from __future__ import unicode_literals # 要放置顶部
import jieba  # pip install jieba

'''
jieba中文处理：【ps:结巴的意思，就是把句子分成多个单词，给人一种说话结巴的感觉】
和拉丁语系不同，亚洲语言是不用空格分开每个有意义的词。
当自然语言处理时，通常词汇是对句子和文章理解的基础，因此需要一个工具去把完整的文本中分解成粒度更细的词。

jieba就是这样一个非常好用的中文工具，是以分词起家的，但是功能比分词要强大很多。

命令行/terminal
    打开文件：python -m jieba news.txt > cut_result.txt
    查看帮助：python -m jieba --help

参考资料：
    github:https://github.com/fxsjy/jieba
    开源中国地址:http://www.oschina.net/p/jieba/?fromerr=LRXZzk9z

使用python做NLP时经常会遇到文本的编码、解码问题，很常见的一种解码错误如下，
UnicodeDecodeError: 'gbk' codec can't decode byte 0x88 in position 15: illegal multibyte sequence
解决方法：
    打开中文文本时，设置编码格式如：open(r'1.txt',encoding='gbk')；
    如果报错，可能是文本中出现的特殊符号超出了gbk的编码范围，可将'gbk'换成'utf-8'。
    如果还报错，可能是文本中出现的特殊符号超出了utf-8的编码范围，可选择编码范围更广的'gb18030'。
    如果仍然报错，说明文中出现了连'gb18030'也无法编码的字符，可使用'ignore'属性进行忽略，
    如 open(r'1.txt',encoding='gb18030'，errors='ignore')；
    或 open(u'1.txt').read().decode('gb18030','ignore')

python字符串前加u或r

    在没有声明编码方式时，默认ASCI编码。如果要指定编码方式，可在文件顶部加入类似如下代码：
    # -*- coding: utf-8 -*-
    或
    # -*- coding: cp936 -*-
    utf-8、cp936是两种编码方式，都支持中文，当然还有其他的编码方式，如gb2312等。

    u/U:python2中，带u表示unicode string，使用unicode进行编码，没有u则表示byte string,类型是str，
    一般英文字符在使用各种编码下, 基本都可正常解析, 所以一般不带u；
    但中文, 必须表明所需编码, 否则一旦编码转换就会出现乱码。建议所有编码方式采用utf8

    r/R:非转义的原始字符串
    字母前加r表示raw string，与特数字符的escape规则有关，一般在正则表达式里面。
    以r开头的字符，常用于正则表达式，对应着re模块。
    特例，'r'可避免字符转义，如果字符串中包含转义字符，不加'r'会被转义，而加了'r'之后就能保留。
    例如：print('abc\n') => abc      print(r'abc\n') => r'abc\n'

    r和u可以搭配使用，例如ur"abc"。

    b:bytes
    python3.x里默认的str是(py2.x里的)unicode, bytes是(py2.x)的str, b”“前缀代表的就是bytes
    python2.x里, b前缀没什么具体意义， 只是为了兼容python3.x的这种写法
'''
'''
1.基本分词函数与用法
jieba.cut( wordString , cut_all=False , HMM=False ) 函数:
    wordString 需要分词的字符串
    cut_all 参数用来控制是否采用全模式
    HMM 是否使用 HMM 模型
    返回一个可迭代的 generator 结构的分词，可用for循环得到的每一个词语(unicode)

jieba.cut_for_search( wordString , HMM=False ) 函数:
    wordString 需要分词的字符串
    HMM 是否使用 HMM 模型
    返回一个可迭代的 generator 结构的分词，可用for循环得到的每一个词语(unicode)
    该方法适合用于搜索引擎构建倒排索引的分词，粒度比较细
'''
seg_list_0 = jieba.cut("我在学习自然语言处理", cut_all=True)
seg_list_1 = jieba.cut("我在学习自然语言处理", cut_all=False)
seg_list_1_1 = jieba.cut("我在学习自然语言处理", cut_all=False, HMM=False)
seg_list_1_2 = jieba.cut("我在学习自然语言处理", cut_all=False, HMM=True)
seg_list_1_3 = jieba.cut("我在学习自然语言处理", cut_all=True, HMM=False)
seg_list_1_4 = jieba.cut("我在学习自然语言处理", cut_all=True, HMM=True)
print (seg_list_0)
print("Full Mode : " + "/ ".join(seg_list_0))  # 全模式
print("Default Mode: " + "/ ".join(seg_list_1))  # 精确模式
# print("cut_all=False, HMM=False : " + "/ ".join(seg_list_1_1))
# print("cut_all=False, HMM=True : " + "/ ".join(seg_list_1_2))
# print("cut_all=True, HMM=False : " + "/ ".join(seg_list_1_3))
# print("cut_all=True, HMM=True : " + "/ ".join(seg_list_1_4))
'''
Building prefix dict from the default dictionary ...
<generator object Tokenizer.cut at 0x0000000003561FC0>
Loading model from cache C:\\Users\\ADMINI~1\AppData\Local\Temp\jieba.cache

Full Mode:    我/ 在/ 学习/ 自然/ 自然语言/ 语言/ 处理
Default Mode: 我/ 在/ 学习/ 自然语言/ 处理
'''

seg_list_2 = jieba.cut("小明硕士毕业于中国科学院计算所，后在哈佛大学深造")  # 默认是精确模式
seg_list_3 = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在哈佛大学深造")  # 搜索引擎模式
print(", ".join(seg_list_2))
print(", ".join(seg_list_3))
'''
小明, 硕士, 毕业, 于, 中国科学院, 计算所, ，, 后, 在, 哈佛大学, 深造
小明, 硕士, 毕业, 于, 中国, 科学, 学院, 科学院, 中国科学院, 计算, 计算所, ，, 后, 在, 哈佛, 大学, 哈佛大学, 深造

### jieba.lcut以及jieba.lcut_for_search 直接返回 list
'''
result_lcut = jieba.lcut("小明硕士毕业于中国科学院计算所，后在哈佛大学深造")
print (result_lcut)
print (" ".join(result_lcut))
print (" ".join(jieba.lcut_for_search("小明硕士毕业于中国科学院计算所，后在哈佛大学深造")))
'''
['小明', '硕士', '毕业', '于', '中国科学院', '计算所', '，', '后', '在', '哈佛大学', '深造']
小明 硕士 毕业 于 中国科学院 计算所 ， 后 在 哈佛大学 深造
小明 硕士 毕业 于 中国 科学 学院 科学院 中国科学院 计算 计算所 ， 后 在 哈佛 大学 哈佛大学 深造
'''

'''
### 添加用户自定义词典
很多时候需要针对不同场景进行分词，会有一些领域内的专有词汇。
    1.可用jieba.load_userdict(file_name)加载用户字典
    2.少量的词汇可用下面方法，直接手动添加：
用 add_word(word, freq=None, tag=None) 和 del_word(word) 在程序中动态修改词典
用 suggest_freq(segment, tune=True) 可调节单个词语的词频，使其能（或不能）被分出来。
'''
print('/'.join(jieba.cut('如果放到旧字典中将出错。', HMM=False)))
jieba.suggest_freq(('中', '将'), True) #添加了分词，就相当于把 中将 一个词分成了两个词
print('/'.join(jieba.cut('如果放到旧字典中将出错。', HMM=False)))
'''
如果/放到/旧/字典/中将/出错/。
如果/放到/旧/字典/中/将/出错/。
'''

print ("---------------------我是分割线----------------")

'''
关键词提取，基于 TF-IDF 算法的关键词抽取
import jieba.analyse
jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=())
    sentence 为待提取的文本
    topK 为返回几个 TF/IDF 权重最大的关键词，默认值为 20
    withWeight 为是否一并返回关键词权重值，默认值为 False
    allowPOS 仅包括指定词性的词，默认值为空，即不筛选
'''
import jieba.analyse as analyse
lines = open(r'../0-common_dataSet/NBA.txt',encoding='utf-8').read()
lines2 = open(u'../0-common_dataSet/西游记.txt',encoding='gb18030').read()
print ("  ".join(analyse.extract_tags(lines, topK=20, withWeight=False, allowPOS=())))
print ("  ".join(analyse.extract_tags(lines2, topK=20, withWeight=False, allowPOS=())))
'''
韦少  杜兰特  全明星  全明星赛  MVP  威少  正赛  科尔  投篮  勇士  球员  斯布鲁克  更衣柜  张卫平  NBA  三连庄  西部  指导  雷霆  明星队
行者  八戒  师父  三藏  唐僧  大圣  沙僧  妖精  菩萨  和尚  那怪  那里  长老  呆子  徒弟  怎么  不知  老孙  国王  一个
'''

print ("---------------------我是分割线----------------")

'''
关于TF-IDF 算法的关键词抽取补充

关键词提取所使用逆向文件频率（IDF）文本语料库可以切换成自定义语料库的路径
    用法：jieba.analyse.set_idf_path(file_name) # file_name为自定义语料库的路径

自定义语料库示例见这里
用法示例见这里关键词提取所使用停止词（Stop Words）文本语料库可以切换成自定义语料库的路径
    用法：jieba.analyse.set_stop_words(file_name) # file_name为自定义语料库的路径

自定义语料库示例见这里
    用法示例见这里

关键词一并返回关键词权重值示例
    用法示例见这里

基于 TextRank 算法的关键词抽取
    https://blog.csdn.net/kamendula/article/details/51756552

'''



'''
jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
#直接使用，接口相同，注意默认过滤词性。
jieba.analyse.TextRank() #新建自定义 TextRank 实例
算法论文：TextRank: Bringing Order into Texts
基本思想:
    将待抽取关键词的文本进行分词
    以固定窗口大小(默认为5，通过span属性调整)，词之间的共现关系，构建图
    计算图中节点的PageRank，注意是无向带权图
'''
import jieba.analyse as analyse
lines = open(r'../0-common_dataSet/NBA.txt',encoding='utf-8').read()
print ("  ".join(analyse.textrank(lines, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))))
print ("  ".join(analyse.textrank(lines, topK=20, withWeight=False, allowPOS=('ns', 'n'))))
'''
全明星赛  勇士  正赛  指导  对方  投篮  球员  没有  出现  时间  威少  认为  看来  结果  相隔  助攻  现场  三连庄  介绍  嘉宾
勇士  正赛  全明星赛  指导  投篮  玩命  时间  对方  现场  结果  球员  嘉宾  时候  全队  主持人  特点  大伙  肥皂剧  全程  快船队
'''
lines22 = open(u'../0-common_dataSet/西游记.txt',encoding='gb18030').read()
print ("  ".join(analyse.textrank(lines22, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))))
'''
行者  师父  八戒  三藏  大圣  不知  菩萨  妖精  只见  长老  国王  却说  呆子  徒弟  小妖  出来  不得  不见  不能  师徒
'''

print ("---------------------我是分割线----------------")

'''
### 词性标注
jieba.posseg.POSTokenizer(tokenizer=None) 新建自定义分词器，
tokenizer 参数可指定内部使用的
jieba.Tokenizer 分词器。jieba.posseg.dt 为默认词性标注分词器。
标注句子分词后每个词的词性，采用和 ictclas 兼容的标记法。
具体的词性对照表参见计算所汉语词性标记集
'''
import jieba.posseg as pseg
words = pseg.cut("我爱自然语言处理")
for word, flag in words:
    print('%s %s' % (word, flag))
    '''
    我 r
    爱 v
    自然语言 l
    处理 v
    '''

# pt_1 = pseg.POSTokenizer("我爱自然语言处理")
# print( pt_1 )

print ("---------------------我是分割线----------------")
'''
并行分词
原理：将目标文本按行分隔后，把各行文本分配到多个 Python 进程并行分词，再归并结果，从而获得分词速度的可观提升
基于 python 自带的 multiprocessing 模块，目前暂不支持 Windows
用法：
jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数
jieba.disable_parallel() # 关闭并行分词模式
实验结果：在 4 核 3.4GHz Linux 机器上，对金庸全集进行精确分词，获得了 1MB/s 的速度，是单进程版的 3.3 倍。
注意：并行分词仅支持默认分词器 jieba.dt 和 jieba.posseg.dt。
'''
import sys
import time
import jieba

'''
jieba.enable_parallel()
NotImplementedError: jieba: parallel mode only supports posix system
                     jieba: 并行模式只支持posix系统
POSIX 可移植操作系统接口（Portable Operating System Interface of UNIX，缩写为 POSIX ）详情百度百科
如下注释的代码，暂不能在window下运行
'''
# jieba.enable_parallel()
# content = open(u'../0-common_dataSet/西游记.txt',"r").read()
# t1 = time.time()
# words = "/ ".join(jieba.cut(content))
# t2 = time.time()
# tm_cost = t2-t1
# print('并行分词速度为 %s bytes/second' % (len(content)/tm_cost))
#
# jieba.disable_parallel()
# content = open(u'../0-common_dataSet/西游记.txt',"r").read()
# t1 = time.time()
# words = "/ ".join(jieba.cut(content))
# t2 = time.time()
# tm_cost = t2-t1
# print('非并行分词速度为 %s bytes/second' % (len(content)/tm_cost))
'''
并行分词速度为 830619.50933 bytes/second
非并行分词速度为 259941.448353 bytes/second
'''

print ("---------------------我是分割线----------------")

'''
Tokenize：返回词语在原文的起止位置
注意，输入参数只接受 unicode
'''
print ("这是默认模式的tokenize")
result = jieba.tokenize(u'自然语言处理非常有用')
for tk in result:
    print("%s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))

print ("这是搜索模式的tokenize")
result = jieba.tokenize(u'自然语言处理非常有用', mode='search')
for tk in result:
    print("%s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))
'''
这是默认模式的tokenize
自然语言		 start: 0 		 end:4
处理		 start: 4 		 end:6
非常		 start: 6 		 end:8
有用		 start: 8 		 end:10

这是搜索模式的tokenize
自然		 start: 0 		 end:2
语言		 start: 2 		 end:4
自然语言		 start: 0 		 end:4
处理		 start: 4 		 end:6
非常		 start: 6 		 end:8
有用		 start: 8 		 end:10
'''

print ("---------------------我是分割线----------------")

# -*- coding: UTF-8 -*-
import sys,os
sys.path.append("../")
from whoosh.index import create_in,open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
'''
ChineseAnalyzer for Whoosh 搜索引擎
from jieba.analyse import ChineseAnalyzer

Whoosh是一个索引文本和搜索文本的类库，可提供搜索文本的服务，
比如创建一个博客的软件，可用whoosh为它添加一个搜索功能以便用户来搜索博客的入口
pip install whoosh 即可安装
'''

analyzer = jieba.analyse.ChineseAnalyzer()
schema = Schema(title=TEXT(stored=True),
                path=ID(stored=True),
                content=TEXT(stored=True, analyzer=analyzer))

if not os.path.exists("tmp"):
    os.mkdir("tmp")

ix = create_in("tmp", schema) # for create new index
#ix = open_dir("tmp") # for read only
writer = ix.writer()

writer.add_document(
    title="document1",
    path="/a",
    content="This is the first document we've added!")

writer.add_document(
    title="document2",
    path="/b",
    content="The second one 你 中文测试中文 is even more interesting! 吃水果")

writer.add_document(
    title="document3",
    path="/c",
    content="买水果然后来世博园。")

writer.add_document(
    title="document4",
    path="/c",
    content="工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作")

writer.add_document(
    title="document4",
    path="/c",
    content="咱俩交换一下吧。")

writer.commit()
searcher = ix.searcher()
parser = QueryParser("content", schema=ix.schema)

for keyword in ("水果世博园","你","first","中文","交换机","交换"):
    print(keyword+"的结果为如下：")
    q = parser.parse(keyword)
    results = searcher.search(q)
    for hit in results:
        print(hit.highlights("content"))

    print("\n--------------for循环--------------\n")

for t in analyzer("我的好朋友是李明;我爱北京天安门;IBM和Microsoft; I have a dream. this is intetesting and interested me a lot"):
    print(t.text)
'''
水果世博园的结果为如下：
买<b class="match term0">水果</b>然后来<b class="match term1">世博园</b>

--------------for循环--------------

你的结果为如下：
second one <b class="match term0">你</b> 中文测试中文 is even more interesting

--------------for循环--------------

first的结果为如下：
<b class="match term0">first</b> document we've added

--------------for循环--------------

中文的结果为如下：
second one 你 <b class="match term0">中文</b>测试<b class="match term0">中文</b> is even more interesting

--------------for循环--------------

交换机的结果为如下：
干事每月经过下属科室都要亲口交代24口<b class="match term0">交换机</b>等技术性器件的安装工作

--------------for循环--------------

交换的结果为如下：
咱俩<b class="match term0">交换</b>一下吧
干事每月经过下属科室都要亲口交代24口<b class="match term0">交换</b>机等技术性器件的安装工作

--------------for循环--------------

我
好
朋友
是
李明
我
爱
北京
天安
天安门
ibm
microsoft
dream
intetest
interest
me
lot
'''