import re , sklearn
print(sklearn.__version__)  # 查看 包的版本号
from sklearn.model_selection import train_test_split
'''
ImportError: No module named 'sklearn.model_selection'
原因是 sklearn 版本过低(低于0.19.1)，在 terminal 输入命令 conda update scikit-learn 更新即可
conda list 查看版本
'''
from sklearn.feature_extraction.text import CountVectorizer
'''
https://blog.csdn.net/m0_37324740/article/details/79411651
词频向量化  CountVectorizer 类会将文本中的词语转换为词频矩阵，
如矩阵中包含的元素a[i][j]，表示j词在i类文本下的词频。通过fit_transform函数计算各个词语出现的次数，
通过 get_feature_names() 可获取词袋中所有文本的关键字，通过 toarray()可看到词频矩阵的结果。
'''
from sklearn.naive_bayes import MultinomialNB
'''
用朴素贝叶斯完成语种检测的分类器，准确度还不错。【类似百度翻译中的输入语种的自动检测】
twitter数据，包含 English, French, German, Spanish, Italian, Dutch荷兰语 6种语言。
# 机器学习的算法要取得好效果，数据质量要保证；用正则表达式，去掉噪声数据
'''


'''
规范化，写成一个class
'''
class LanguageDetector():
    def __init__(self, classifier=MultinomialNB()):
        self.classifier = classifier
        '''
        在降噪数据上抽取出来有用的特征，抽取1-gram和2-gram的统计特征
        '''
        self.vectorizer = CountVectorizer(ngram_range=(1, 2),
                                          max_features=1000,
                                          preprocessor=self._remove_noise)

    def _remove_noise(self, document):
        noise_pattern = re.compile("|".join(["http\S+", "\@\w+", "\#\w+"]))
        clean_text = re.sub(noise_pattern, "", document)
        return clean_text

    def features(self, X):
        # print( self.vectorizer.transform(X)[:5] )
        '''
          (0, 134)	1
          (0, 410)	1
        '''
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)

# 打开文件，读取数据存变量后，关闭文件。释放资源
in_f = open(r'../0-common_dataSet/简易语种检测dataSet.csv')
lines = in_f.readlines()
in_f.close()

dataset = [(line.strip()[:-3], line.strip()[-2:]) for line in lines]
x, y = zip(*dataset)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

language_detector = LanguageDetector()
language_detector.fit(x_train, y_train)
print(language_detector.predict('This is an English sentence'))  # ['en']
print(language_detector.score(x_test, y_test))  # 0.977062196736
'''
能在1500句话上，训练得到准确率97.7%的分类器，效果还是不错的。
如果大家加大语料，准确率会非常高。
'''

remove_noise_ = language_detector._remove_noise("Trump images are now more popular than cat gifs. @trump #trends http://www.trumptrends.html")
print('测试降噪数据：',remove_noise_)  # 'Trump images are now more popular than cat gifs.'
