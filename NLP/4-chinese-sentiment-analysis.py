# -*- coding: utf-8 -*-
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC
import sys

'''
python reload(sys)找不到，name 'reload' is not defined
reload(sys)
sys.setdefaultencoding("utf-8")
在Python 3.x中不好使了 提示 name 'reload' is not defined
在3.x中已经被替换为
import importlib
importlib.reload(sys)

sys.setdefaultencoding('utf-8') 报错
AttributeError: module 'sys' has no attribute 'setdefaultencoding'
这种方式在3.x中被彻底遗弃，因为，Python3字符串默认编码unicode, 所以sys.setdefaultencoding已经不存在了。
'''
import importlib
importlib.reload(sys)

'''
载入所需的库
我们依旧会用gensim去做word2vec的处理，会用sklearn当中的SVM进行建模
/Library/Python/2.7/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
  "This module will be removed in 0.20.", DeprecationWarning)
'''
# 取数据，处理后  return  x_train , x_test
def load_file_and_preprocessing():
    neg=pd.read_excel('data/neg.xls',header=None,index=None)
    pos=pd.read_excel('data/pos.xls',header=None,index=None)

    cw = lambda x: list(jieba.cut(x))
    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)

    # print (pos['words'])
    '''
    0        [做, 父母, 一定, 要, 有, 刘墉, 这样, 的, 心态, ，, 不断, 地, 学习,...
    1        [作者, 真有, 英国人, 严谨, 的, 风格, ，, 提出, 观点, 、, 进行, 论述,...
    2        [作者, 长篇大论, 借用, 详细, 报告, 数据处理, 工作, 和, 计算结果, 支持, ...
    3        [作者, 在, 战, 几时, 之前, 用, 了, ＂, 拥抱, ＂, 令人, 叫绝, ．, ...
    4        [作者, 在, 少年, 时即, 喜, 阅读, ，, 能, 看出, 他, 精读, 了, 无数,...
             .......
    '''
    # use 1 for positive sentiment, 0 for negative
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
    # print(y)  # [1. 1. 1. ... 0. 0. 0.]

    x_train, x_test, y_train, y_test = \
        train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)

    np.save('svm_data/y_train.npy',y_train)
    np.save('svm_data/y_test.npy',y_test)
    return x_train,x_test

# 对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(text, size,imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

#计算词向量
def get_train_vecs(x_train,x_test):
    print(x_train)
    print(x_test.shape)
    ''' 竟是这样的数据
    [list(['志玲', '姐姐'
     list(['1', '.',
     list(['现场', '看到'
     ...
     list(['可能', '是',
     list(['值得', '推荐'])]
    '''
    # x_train,x_test 是 'numpy.ndarray' object，没有 .head() 属性

    n_dim = 300  # 通常是 50 - 300 维 的 向量，超过300的，较少。
    #初始化模型和词表
    imdb_w2v = Word2Vec(size=n_dim, min_count=10)
    help(imdb_w2v)
    imdb_w2v.build_vocab(x_train)

    #在评论训练集上建模(可能会花费几分钟)
    imdb_w2v.train(x_train)

    train_vecs = np.concatenate([build_sentence_vector(z, n_dim,imdb_w2v) for z in x_train])
    #train_vecs = scale(train_vecs)

    np.save('svm_data/train_vecs.npy',train_vecs)
    print (train_vecs.shape)
    #在测试集上训练
    imdb_w2v.train(x_test)
    imdb_w2v.save('svm_data/w2v_model/w2v_model.pkl')
    #Build test tweet vectors then scale
    test_vecs = np.concatenate([build_sentence_vector(z, n_dim,imdb_w2v) for z in x_test])
    #test_vecs = scale(test_vecs)
    np.save('svm_data/test_vecs.npy',test_vecs)
    print (test_vecs.shape)

def get_data():
    train_vecs=np.load('svm_data/train_vecs.npy')
    y_train=np.load('svm_data/y_train.npy')
    test_vecs=np.load('svm_data/test_vecs.npy')
    y_test=np.load('svm_data/y_test.npy')
    return train_vecs,y_train,test_vecs,y_test

def svm_train(train_vecs,y_train,test_vecs,y_test):
    clf=SVC(kernel='rbf',verbose=True)
    clf.fit(train_vecs,y_train)
    joblib.dump(clf, 'svm_data/svm_model/model.pkl')
    print (clf.score(test_vecs,y_test))

'''
训练模型
'''
x_train,x_test = load_file_and_preprocessing()
get_train_vecs(x_train,x_test)
train_vecs,y_train,test_vecs,y_test = get_data()
svm_train(train_vecs,y_train,test_vecs,y_test)



'''
构建待预测句子的向量
'''
def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load('svm_data/w2v_model/w2v_model.pkl')
    #imdb_w2v.train(words)
    train_vecs = build_sentence_vector(words, n_dim,imdb_w2v)
    #print train_vecs.shape
    return train_vecs

'''
对单个句子进行情感判断
'''
def svm_predict(string):
    words=jieba.lcut(string)
    words_vecs=get_predict_vecs(words)
    clf=joblib.load('svm_data/svm_model/model.pkl')

    result=clf.predict(words_vecs)

    if int(result[0])==1:
        print (string,' positive')
    else:
        print (string,' negative')

##对输入句子情感进行判断
string='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
#string='牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
svm_predict(string)

