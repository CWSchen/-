import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer

'''
关键词搜索  search + relevance(相关性,关联,关联性)
Kaggle竞赛题：https://www.kaggle.com/c/home-depot-product-search-relevance
本篇的教程里会尽量用点不一样的库，让大家感受一下Python NLP领域各个库的优缺点。
Step1：导入所需库
'''
df_train = pd.read_csv('input/train.csv.zip', encoding="ISO-8859-1")
df_test = pd.read_csv('input/test.csv.zip', encoding="ISO-8859-1")
print( df_train.head() )

df_desc = pd.read_csv('input/product_descriptions.csv.zip')
print( df_desc.head() )
'''
不要做太多的复杂处理，直接合并测试/训练集，以便于统一做进一步的文本预处理
'''
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
print(df_all.head() , '\n' , df_all.shape)
'''
产品介绍也是一个极有用的信息，我们把它拿过来：
'''
df_all = pd.merge(df_all, df_desc, how='left', on='product_uid')
print( df_all.head() )
'''
好了，现在我们得到一个全体的数据大表格

Step 2: 文本预处理
我们这里遇到的文本预处理比较简单，因为最主要的就是看关键词是否会被包含。
所以我们统一化我们的文本内容，以达到任何term在我们的数据集中只有一种表达式的效果。
我们这里用简单的Stem做个例子：
（有兴趣的同学可以选用各种你觉得靠谱的预处理方式：去掉停止词，纠正拼写，去掉数字，去掉各种emoji，等等）
'''
stemmer = SnowballStemmer('english')

def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])
'''
为了计算『关键词』的有效性，我们可以naive地直接看『出现了多少次』
'''
def str_common_word(str1, str2):
    return sum(int(str2.find(word)>=0) for word in str1.split())
'''
接下来，把每一个column都跑一遍，以清洁所有的文本内容
'''
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))

df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))

df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))
'''
Step 3: 自制文本特征
一般属于一种脑洞大开的过程，想到什么可以加什么。
当然，特征也不是越丰富越好，稍微靠谱点是肯定的。
关键词的长度：
'''
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
'''
标题中有多少关键词重合
'''
df_all['commons_in_title'] = df_all.apply(lambda x:str_common_word(x['search_term'],x['product_title']), axis=1)
'''
描述中有多少关键词重合
'''
df_all['commons_in_desc'] = df_all.apply(lambda x:str_common_word(x['search_term'],x['product_description']), axis=1)
'''
等等等等。。变着法子想出些数字能代表的features，一股脑放进来~
搞完之后，我们把不能被『机器学习模型』处理的column给drop掉
'''
df_all = df_all.drop(['search_term','product_title','product_description'],axis=1)
'''
Step 4: 重塑训练/测试集
舒淇说得好，要把之前脱下的衣服再一件件穿回来
数据处理也是如此，搞完一圈预处理之后，我们让数据重回原本的样貌
分开训练和测试集
'''
df_train = df_all.loc[df_train.index]
df_test = df_all.loc[df_test.index]
'''
记录下测试集的id
留着上传的时候 能对的上号
'''
test_ids = df_test['id']
'''
分离出y_train
'''
y_train = df_train['relevance'].values
'''
把原集中的label给删去
否则就是cheating了
'''
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values
'''
Step 5: 建立模型
我们用个最简单的模型：Ridge回归模型
'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# 用CV结果保证公正客观性；并调试不同的alpha值
params = [1,3,5,6,7,8,9,10]
test_scores = []
for param in params:
    clf = RandomForestRegressor(n_estimators=30, max_depth=param)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))


#matplotlib 查看 Param 取何值时， CV Error 有最优解
import matplotlib.pyplot as plt
plt.plot(params, test_scores)
plt.title("Param vs CV Error")
plt.show() # 大概6~7的时候达到了最优解


'''
Step 6: 上传结果
用我们测试出的最优解建立模型，并跑跑测试集
'''
rf = RandomForestRegressor(n_estimators=30, max_depth=6)
rf.fit(X_train, y_train)
# RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=6,
#            max_features='auto', max_leaf_nodes=None,
#            min_impurity_split=1e-07, min_samples_leaf=1,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            n_estimators=30, n_jobs=1, oob_score=False, random_state=None,
#            verbose=0, warm_start=False)
y_pred = rf.predict(X_test)

# 把拿到的结果，放进PD，做成CSV上传：
pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv('submission.csv',index=False)
'''
总结：
这一篇教程中，虽然都是用的最简单的方法，但是基本框架是很完整的。
可尝试修改/调试/升级的部分是：
文本预处理步骤: 你可以使用很多不同的方法来使得文本数据变得更加清洁
自制的特征: 相处更多的特征值表达方法（关键词全段重合数量，重合比率，等等）
更好的回归模型: 根据之前的课讲的Ensemble方法，把分类器提升到极致
'''