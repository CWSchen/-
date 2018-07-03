import numpy as np
import pandas as pd
import re

'''
LDA模型应用：一眼看穿希拉里的邮件
我们拿到希拉里泄露的邮件，跑一把LDA，看看她平时都在聊什么。
'''
df = pd.read_csv("input/HillaryEmails.csv") # 总行数 382920

#'Id','ExtractedBodyText' 是文档中的两个特征/列，其中数据中有很多Nan的值，直接移除。
df = df[['Id','ExtractedBodyText']].dropna()

#文本预处理：(对NLP是很重要的)针对邮件内容，写一组正则表达式
def clean_email_text(text):
    text = text.replace('\n'," ") #新行，不需要的
    text = re.sub(r"-", " ", text) #把 "-" 的两个单词，分开。（比如：july-edu ==> july edu）
    text = re.sub(r"\d+/\d+/\d+", "", text) #日期，对主体模型没意义 - 移除
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text) #时间，没意义 - 移除
    text = re.sub(r"[\w]+@[\.\w]+", "", text) #邮件地址，没意义 - 移除
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text) #网址，没意义 - 移除
    pure_text = ''
    '''以防还有其他特殊字符（数字）等等，我们直接把他们loop一遍，过滤掉'''
    for letter in text:
        '''只留下字母和空格'''
        if letter.isalpha() or letter==' ':
            pure_text += letter
    '''再把那些去除特殊字符后 落单/仅一个 的单词移除。只剩下有意义的单词'''
    text = ' '.join(word for word in pure_text.split() if len(word)>1)
    return text
    
    
docs = df['ExtractedBodyText']
# print('df[\"ExtractedBodyText\"] \n',docs)
docs = docs.apply(lambda s: clean_email_text(s))
print( 'docs.head(1).values = ' , docs.head(1).values )
doclist = docs.values
print( 'doclist = ' , doclist )


'''
LDA 模型构建：
好，我们用 Gensim 来做一次模型构建
首先，我们得把我们刚刚整出来的一大波文本数据
[[一条邮件字符串]，[另一条邮件字符串], ...]
转化成Gensim认可的语料库形式：
[[一，条，邮件，在，这里],[第，二，条，邮件，在，这里],[今天，天气，肿么，样],...]
引入库：
'''
from gensim import corpora, models, similarities
import gensim   # pip install gensim
'''
http://www.cnblogs.com/iloveai/p/gensim_tutorial.html
Gensim是一款开源的第三方Python工具包，用于从原始的非结构化的文本中，无监督地学习到文本隐层的主题向量表达。
它支持包括 TF-IDF，LSA，LDA，和 word2vec 在内的多种主题模型算法，支持流式训练，
并提供了诸如相似度计算，信息检索等一些常用任务的API接口。
'''

'''
为了免去讲解安装NLTK等等的麻烦，我这里直接手写一下停止词列表：
这些词在不同语境中指代意义完全不同，但是在不同主题中的出现概率是几乎一致的。所以要去除，否则对模型的准确性有影响
'''
# stoplist = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours',
#             'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their',
#             'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once',
#             'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you',
#             'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will',
#             'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be',
#             'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself',
#             'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both',
#             'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn',
#             'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about',
#             'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn',
#             'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']

stoplist = [ line.rstrip() for line in open('./input/stopwords.txt')]
print('stoplist \n',stoplist)
'''
人工分词：(英文分词，直接空格分割即可)
中文的分词稍微复杂点儿，具体可以百度：CoreNLP, HaNLP, 结巴分词，等等
分词的意义在于，把原文本中的长字符串，分割成有意义的小元素。
'''
texts = [[word for word in doc.lower().split() if word not in stoplist] for doc in doclist]
print( texts[0] )

#建立语料库。用词袋的方法，把每个单词用一个数字 index 指代，并把原文本变成一条长长的数组：
dictionary = corpora.Dictionary(texts)   # corpora / corpus  语料库,全集,复数
corpus = [dictionary.doc2bow(text) for text in texts] # bow 全称 bag of word
print( 'corpus[13]\n' , corpus[13] )

'''
这个列表告诉我们，第14（从0开始是第一）个邮件中，一共6个有意义的单词（经过我们的文本预处理，并去除了停止词后）
其中，36号单词出现1次，505号单词出现1次，以此类推。。。

接着，我们终于可以建立模型了：
'''
# lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20) # 老版本
lda = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20) # 新版本

print( '第10号分类，其中最常出现的单词\n',lda.print_topic(10, topn=5) )
print( '把所有的主题打印\n',lda.print_topics(num_topics=20, num_words=5) )
'''
接下来：
通过 lda.get_document_topics(bow) 或 lda.get_term_topics(word_id)
两个方法，把新鲜的文本/单词，分类成20个主题中的一个。

但是注意，我们这里的文本和单词，都必须得经过同样步骤的文本预处理+词袋化，也就是说，变成数字表示每个单词的形式。
'''

