from gensim.models import word2vec

sentences = word2vec.Text8Corpus("C:/traindataw2v.txt")  # 加载语料
model = word2vec.Word2Vec(sentences, size=200)  # 训练skip-gram模型; 默认window=5
# 获取“学习”的词向量
print("学习：" + model["学习"])
# 计算两个词的相似度/相关程度
y1 = model.similarity("不错", "好")
# 计算某个词的相关词列表
y2 = model.most_similar("书", topn=20)  # 20个最相关的
# 寻找对应关系
print("书-不错，质量-")
y3 = model.most_similar(['质量', '不错'], ['书'], topn=3)
# 寻找不合群的词
y4 = model.doesnt_match("书 书籍 教材 很".split())
# 保存模型，以便重用
model.save("db.model")
# 对应的加载方式
model = word2vec.Word2Vec.load("db.model")
'''
作者：夜尽天明时
链接：https://www.jianshu.com/p/972d0db609f2
來源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
'''