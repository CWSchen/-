import os
import re
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import nltk
#nltk.download()
# from nltk.corpus import stopwords

# datafile = os.path.join('..', 'data', 'labeledTrainData.tsv')
datafile = os.path.join('.', 'data', 'labeledTrainData.tsv')
# print(datafile) # .\data\labeledTrainData.tsv
df = pd.read_csv(datafile, sep='\t', escapechar='\\')
print('Number of reviews: {}'.format(len(df)))
df.head()

'''
对影评数据做预处理，大概有以下环节：
去掉html标签
移除标点
切分成词/token
去掉停用词
重组为新的句子
'''

def display(text, title):
    print(title,'\n',text,'\n----------我是分割线-------------')


raw_example = df['review'][1]
display(raw_example, '原始数据')


example = BeautifulSoup(raw_example, 'html.parser').get_text()
display(example, '去掉HTML标签的数据')


example_letters = re.sub(r'[^a-zA-Z]', ' ', example)
display(example_letters, '去掉标点的数据')


words = example_letters.lower().split()
display(words, '纯词列表数据')


'''
#下载停用词和其他语料会用到
#nltk.download()
'''

#words_nostop = [w for w in words if w not in stopwords.words('english')]
'''
使用下载下来的停用词库
'''
stopwords = {}.fromkeys([ line.rstrip() for line in open('./data/stopwords.txt')])
display(stopwords, '停用词数据')
words_nostop = [w for w in words if w not in stopwords]
display(words_nostop, '去掉停用词数据')

#eng_stopwords = set(stopwords.words('english'))
eng_stopwords = set(stopwords)

def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [w for w in words if w not in eng_stopwords]
    return ' '.join(words)


'''
清洗数据添加到 dataframe 里
'''
df['clean_review'] = df.review.apply(clean_text)
df.head()


'''
抽取 bag of words 特征(用sklearn的CountVectorizer)
'''
vectorizer = CountVectorizer(max_features = 5000)
train_data_features = vectorizer.fit_transform(df.clean_review).toarray()
print(train_data_features.shape)
'''
/usr/local/lib/python2.7/site-packages/numpy/core/fromnumeric.py:2652: VisibleDeprecationWarning: `rank` is deprecated; use the `ndim` attribute or function instead. To find the rank of a matrix see `numpy.linalg.matrix_rank`.
  VisibleDeprecationWarning)
Out[13]:
(25000, 5000)

训练分类器
'''
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_features, df.sentiment)
'''
在训练集上做个predict看看效果如何
'''
print('*'*100)
display(confusion_matrix(df.sentiment, forest.predict(train_data_features)),'混淆矩阵')
print('*'*100)
'''
混淆矩阵是除了ROC曲线和AUC之外的另一个判断分类好坏程度的方法。详情百度。

混淆矩阵(Confusion Matrix):的每一列代表了预测类别 ，
每一列的总数表示预测为该类别的数据的数目;每一行代表了数据的真实归属类别 ，
每一行的数据总数表示该类别的数据实例的数目。每一列中的数值表示真实数据被预测为该类的数目:
如下图，第一行第一列中的43表示有43个实际归属第一类的实例被预测为第一类，
同理，第二行第一列的2表示有2个实际归属为第二类的实例被错误预测为第一类。
'''
'''
array([[12500,     0],
       [    0, 12500]])

删除不用的占内容变量
'''

del df
del train_data_features
'''
读取测试数据进行预测
'''

datafile = os.path.join('.', 'data', 'testData.tsv')
df = pd.read_csv(datafile, sep='\t', escapechar='\\')
print('Number of reviews: {}'.format(len(df)))
df['clean_review'] = df.review.apply(clean_text)
df.head()
'''
Number of reviews: 25000
'''

test_data_features = vectorizer.transform(df.clean_review).toarray()
print(test_data_features.shape)

result = forest.predict(test_data_features)
output = pd.DataFrame({'id':df.id, 'sentiment':result})


output.head()


# output.to_csv(os.path.join('..', 'data', 'Bag_of_Words_model.csv'), index=False)
output.to_csv(os.path.join('.', 'data', 'Bag_of_Words_model.csv'), index=False)

del df
del test_data_features