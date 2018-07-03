import nltk
import math
import string
from nltk.corpus import stopwords  # 停词
from collections import Counter
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer

'''
TF-IDF算法解析与Python实现
来源：https://blog.csdn.net/baimafujinji/article/details/51476117

TF-IDF（term frequency–inverse document frequency）
是一种用于信息检索（information retrieval）与文本挖掘（text mining）的常用加权技术。
比较容易理解的一个应用场景是，有一些文章，希望计算机能够自动地进行关键词提取。

TF-IDF就是可以完成这项任务的一种统计方法。它能够用于评估一个词语对于一个文集或一个语料库中的其中一份文档的重要程度。
'''
# 用作处理对象的三段文本
text1 = "Python is a 2000 made-for-TV horror movie directed by Richard \
Clabaugh. The film features several cult favorite actors, including William \
Zabka of The Karate Kid fame, Wil Wheaton, Casper Van Dien, Jenny McCarthy, \
Keith Coogan, Robert Englund (best known for his role as Freddy Krueger in the \
A Nightmare on Elm Street series of films), Dana Barron, David Bowe, and Sean \
Whalen. The film concerns a genetically engineered snake, a python, that \
escapes and unleashes itself on a small town. It includes the classic final\
girl scenario evident in films like Friday the 13th. It was filmed in Los Angeles, \
 California and Malibu, California. Python was followed by two sequels: Python \
 II (2002) and Boa vs. Python (2004), both also made-for-TV films."

text2 = "Python, from the Greek word (πύθων/πύθωνας), is a genus of \
nonvenomous pythons[2] found in Africa and Asia. Currently, 7 species are \
recognised.[2] A member of this genus, P. reticulatus, is among the longest \
snakes known."

text3 = "The Colt Python is a .357 Magnum caliber revolver formerly \
manufactured by Colt's Manufacturing Company of Hartford, Connecticut. \
It is sometimes referred to as a \"Combat Magnum\".[1] It was first introduced \
in 1955, the same year as Smith &amp; Wesson's M29 .44 Magnum. The now discontinued \
Colt Python targeted the premium revolver market segment. Some firearm \
collectors and writers such as Jeff Cooper, Ian V. Hogg, Chuck Hawks, Leroy \
Thompson, Renee Smeets and Martin Dougherty have described the Python as the \
finest production revolver ever made."

'''
TF-IDF的基本思想：词语的重要性与它在文件中出现的次数成正比，但同时会随着它在语料库中出现的频率成反比下降。
但无论如何，统计每个单词在文档中出现的次数是必要的操作。所以说，TF-IDF也是一种基于 bag-of-word 的方法。

首先我们来做分词，其中比较值得注意的地方是我们设法剔除了其中的标点符号（显然，标点符号不应该成为最终的关键词）。
'''
def get_tokens(text):
    lowers = text.lower()
    #remove the punctuation using the character deletion step of translate
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    no_punctuation = lowers.translate(remove_punctuation_map)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens

# 去除“词袋'中的“停词'（stop words）,因为像the, a, and 这些词出现次数很多，但与文档所表述的主题是无关的，
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def def_Preprocessing(text):
    # 测试上述分词结果 Counter() 函数用于统计每个单词出现的次数。
    tokens = get_tokens(text)
    count = Counter(tokens)
    print (count.most_common(10)) # 输出出现次数最多的10个词

    tokens = get_tokens(text)
    filtered = [w for w in tokens if not w in stopwords.words('english')]
    count = Counter(filtered)
    print (count.most_common(10))

    '''
    但这个结果还是不太理想，像 films, film, filmed 其实都可以看出是 film，而不应该把每个词型都分别进行统计。
    这时就需要要用 Stemming 方法。
    '''
    tokens = get_tokens(text)
    filtered = [w for w in tokens if not w in stopwords.words('english')]
    stemmer = PorterStemmer()
    stemmed = stem_tokens(filtered, stemmer)

    count = Counter(stemmed) # 输出计数排在前10的词汇（以及它们出现的次数）
    print(count)
    return count

# 至此，就完成了基本的预处理过程
count1 = def_Preprocessing(text1)
count2 = def_Preprocessing(text2)
count3 = def_Preprocessing(text3)

'''
TF-IDF的算法原理

预处理过程，把停词过滤掉。只考虑剩下的有实际意义的词，显然词频（TF，Term Frequency）较高的词对于文章可能更为重要（即潜在的关键词）。
但这样又会遇到了另一个问题，我们可能发现在上面例子中，madefortv、california、includ 都出现了2次
（madefortv其实是原文中的made-for-TV，因为我们所选分词法的缘故，它被当做是一个词来看待），
但这显然并不意味着“作为关键词，它们的重要性是等同的'。

因为'includ'是很常见的词（注意 includ 是 include 的词干）。相比之下，california 可能并不那么常见。
如果这两个词在一篇文章的出现次数一样多，我们有理由认为，california 重要程度要大于 include ，
也就是说，在关键词排序上面，california 应该排在 include 的前面。

于是，我们需要一个重要性权值调整参数，来衡量一个词是不是常见词。如果某个词比较少见，但是它在某篇文章中多次出现，
那么它很可能就反映了这篇文章的特性，它就更有可能揭示这篇文字的话题所在。
这个权重调整参数就是“逆文档频率'（IDF，Inverse Document Frequency），它的大小与一个词的常见程度成反比。

知道了 TF 和 IDF 以后，将这两个值相乘，就得到了一个词的 TF-IDF 值。
某个词对文章的重要性越高，它的TF-IDF值就越大。如果用公式来表示，则对于某个特定文件中的词语 t(i) 而言，
它的 TF 可以表示为：tf(i,j) = N(i,j) / ∑k N(k,j)

其中 n(i,j) 是该词在文件 d(j) 中出现的次数，而分母则是文件 d(j) 中所有词汇出现的次数总和。如果用更直白的表达是来描述就是，
TF(t)= (Number of times term t appears in a document) / (Total number of terms in the document)

某一特定词语的IDF，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取对数即可：
idf(i)=log( |D| / |{ j:t(i)∈d(j) }| )

其中，|D| 是语料库中的文件总数。 |{j:ti∈dj}| 表示包含词语 t(i) 的文件数目（即 n(i,j) != 0 的文件数目）。
如果该词语不在语料库中，就会导致分母为零，因此一般情况下使用 1+|{j:ti∈dj}|
同样，如果用更直白的语言表示就是
IDF(t)= (logeTotal number of documents) / (Number of documents with term t in it)

最后，便可以来计算 TF-IDF(t) = TF(t) * IDF(t)
下面的代码实现了计算TF-IDF值的功能。
'''
def tf(word, count):
    return count[word] / sum(count.values())

def n_containing(word, count_list):
    return sum(1 for count in count_list if word in count)

def idf(word, count_list):
    return math.log(len(count_list) / (1 + n_containing(word, count_list)))

def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)

# 测试代码
countlist = [count1, count2, count3]
for i, count in enumerate(countlist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, count, countlist) for word in count}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))

'''
附：利用Scikit-Learn实现的TF-IDF

因为 TF-IDF 在文本数据挖掘时十分常用，
所以在Python的机器学习包中也提供了内置的TF-IDF实现。主要使用的函数就是 TfidfVectorizer()，来看一个简单的例子。
'''
corpus = ['This is the first document.',
      'This is the second second document.',
      'And the third one.',
      'Is this the first document?',]
vectorizer = TfidfVectorizer(min_df=1)
vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
# ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
print(vectorizer.fit_transform(corpus).toarray())
'''
[[0.         0.43877674 0.54197657 0.43877674 0.         0.
  0.35872874 0.         0.43877674]
 [0.         0.27230147 0.         0.27230147 0.         0.85322574
  0.22262429 0.         0.27230147]
 [0.55280532 0.         0.         0.         0.55280532 0.
  0.28847675 0.55280532 0.        ]
 [0.         0.43877674 0.54197657 0.43877674 0.         0.
  0.35872874 0.         0.43877674]]

最终的结果是一个 4×94×9 矩阵。每行表示一个文档，每列表示该文档中的每个词的评分。
如果某个词没有出现在该文档中，则相应位置就为 0 。数字 9 表示语料库里词汇表中一共有 9 个（不同的）词。
例如，你可以看到在文档1中，并没有出现 and，所以矩阵第一行第一列的值为 0 。
单词 first 只在文档1中出现过，所以第一行中 first 这个词的权重较高。
而 document 和 this 在 3 个文档中出现过，所以它们的权重较低。而 the 在 4 个文档中出现过，所以它的权重最低。

最后需要说明的是，由于函数 TfidfVectorizer() 有很多参数，我们这里仅仅采用了默认的形式，
所以输出的结果可能与采用前面介绍的（最基本最原始的）算法所得出之结果有所差异（但数量的大小关系并不会改变）。
有兴趣的读者可以参考文献[4]来了解更多关于在Scikit-Learn中执行 TF-IDF 算法的细节。
'''