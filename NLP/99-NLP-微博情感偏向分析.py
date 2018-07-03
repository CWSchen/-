import numpy as np
'''
https://blog.csdn.net/baimafujinji/article/details/51051505
自然语言处理（NLP）中一个很重要的研究方向就是语义的情感分析（Sentiment Analysis）。
例如IMDB上有很多关于电影的评论，那么我们就可以通过Sentiment Analysis来评估某部电影的口碑，
（如果它才刚刚上映的话）甚至还可以据此预测它是否能够卖座。
与此相类似，国内的豆瓣上也有很多对影视作品或者书籍的评论内容亦可以作为情感分析的语料库。
对于那些电子商务网站而言，针对某一件商品，我们也可以看到留言区里为数众多的评价内容，
那么同类商品中，哪个产品最受消费者喜爱呢？或许对商品评论的情感分析可以告诉我们答案。
'''

from sklearn.feature_extraction import DictVectorizer
measurements = [  
    {'city': 'Dubai', 'temperature': 33.},  
    {'city': 'London', 'temperature': 12.},  
    {'city': 'San Fransisco', 'temperature': 18.},  
    ]  
vec = DictVectorizer()  
vec.fit_transform(measurements).toarray()  
'''
array([[  1.,   0.,   0.,  33.],
       [  0.,   1.,   0.,  12.],
       [  0.,   0.,   1.,  18.]])
'''
print(vec.get_feature_names())
# ['city=Dubai', 'city=London', 'city=San Fransisco', 'temperature']

measurements = [
    {'city=Dubai': True, 'city=London': True, 'temperature': 33.},
    {'city=London': True, 'city=San Fransisco': True, 'temperature': 12.},
    {'city': 'San Fransisco', 'temperature': 18.},]
print( vec.fit_transform(measurements).toarray()  )
'''
[[ 1.  1.  0. 33.]
 [ 0.  1.  1. 12.]
 [ 0.  0.  1. 18.]]
'''

'''
另外的一个常见问题是训练数据集和测试数据集的字典大小不一致，
此时我们希望短的那个能够通过补零的方式来追平长的那个。这时就需要使用transform。还是来看例子：
'''
D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]  
v = DictVectorizer(sparse=False)  
X = v.fit_transform(D)
print(X)
'''
[[2. 0. 1.]
 [0. 1. 3.]]
'''
print(v.transform({'foo': 4, 'unseen_feature': 3})  )
# [[0. 0. 4.]]
print(v.transform({'foo': 4})  )
# [[0. 0. 4.]]

'''
可见当使用transform之后，后面的那个总是可以实现同前面的一个相同的维度。
当然这种追平可以是补齐，也可以是删减，所以通常，我们都是用补齐短的这样的方式来实现维度一致。
如果你不使用transform，而是继续fit_transform，则会得到下面的结果（这显然不能满足我们的要求）
'''
# 后续的Logistic Regression建立稀疏矩阵
vec = DictVectorizer()
sparse_matrix_tra = vec.fit_transform(feature_dicts_tra)
sparse_matrix_dev = vec.transform(feature_dicts_dev)