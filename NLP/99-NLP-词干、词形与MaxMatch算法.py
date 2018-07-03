#
'''
Python自然语言处理：词干、词形与MaxMatch算法
来源：https://blog.csdn.net/baimafujinji/article/details/51069522

自然语言处理中一个很重要的操作就是所谓的stemming 和 lemmatization，二者非常类似。
它们是词形规范化的两类重要方式，都能够达到有效归并词形的目的，二者既有联系也有区别。

1、词干提取（stemming）
定义：Stemming is the process for reducing inflected (or sometimes derived) words to their stem,
base or root form—generally a written word form.
解释一下，Stemming 是抽取词的词干或词根形式（不一定能够表达完整语义）。
NLTK中提供了三种最常用的词干提取器接口，即 Porter stemmer, Lancaster Stemmer 和 Snowball Stemmer。
Porter Stemmer基于Porter词干提取算法，来看例子
'''

from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()  
porter_stemmer.stem('maximum')  
porter_stemmer.stem('presumably')  
porter_stemmer.stem('multiply')  
porter_stemmer.stem('provision')  
porter_stemmer.stem('owed')

# Lancaster Stemmer 基于Lancaster 词干提取算法，来看例子
from nltk.stem.lancaster import LancasterStemmer  
lancaster_stemmer = LancasterStemmer()  
lancaster_stemmer.stem('maximum')  
lancaster_stemmer.stem('presumably')
lancaster_stemmer.stem('presumably')
lancaster_stemmer.stem('multiply')
lancaster_stemmer.stem('provision')
lancaster_stemmer.stem('owed')

# Snowball Stemmer基于Snowball 词干提取算法，来看例子
from nltk.stem import SnowballStemmer  
snowball_stemmer = SnowballStemmer('english')  
snowball_stemmer.stem('maximum')  
snowball_stemmer.stem('presumably')  
snowball_stemmer.stem('multiply')  
snowball_stemmer.stem('provision')  
snowball_stemmer.stem('owed')

'''
2、词形还原（lemmatization）
定义：Lemmatisation (or lemmatization) in linguistics, is the process of grouping together 
the different inflected forms of a word so they can be analysed as a single item.

可见，Lemmatisation是把一个任何形式的语言词汇还原为一般形式（能表达完整语义）。
相对而言，词干提取是简单的轻量级的词形归并方式，最后获得的结果为词干，并不一定具有实际意义。
词形还原处理相对复杂，获得结果为词的原形，能够承载一定意义，与词干提取相比，更具有研究和应用价值。

我们会在后面给出一个同MaxMatch算法相结合的更为复杂的例子。

3、最大匹配算法（MaxMatch）

MaxMatch算法在中文自然语言处理中常常用来进行分词（或许从名字上你已经能想到它是基于贪婪策略设计的一种算法）。
通常，英语中一句话里的各个词汇之间通过空格来分割，这是非常straightforward的，但是中文却没有这个遍历。
例如“我爱中华人民共和国”，这句话被分词的结果可能是这样的{'我'，'爱'，'中华'，'人民'，'共和国'}，
又或者是{'我'，'爱'，'中华人民共和国'}，显然我们更倾向于后者的分词结果。
因为'中华人民共和国'显然是一个专有名词（把这样一个词分割来看显然并不明智）。
我们选择后者的策略就是所谓的MaxMatch，即最大匹配。因为'中华人民共和国'这个词显然要比'中华'，'人民'，'共和国'这些词都长。

我们可以通过一个英文的例子来演示MaxMatch算法（其实中文处理的道理也是一样的）。
算法从右侧开始逐渐减少字符串长度，以此求得可能匹配的最大长度的字符串。
考虑到我们所获得的词汇可能包含有某种词型的变化，所以其中使用了Lemmatisation，然后在词库里进行匹配查找。
'''

from nltk.stem import WordNetLemmatizer
from nltk.corpus import words

wordlist = set(words.words())
wordnet_lemmatizer = WordNetLemmatizer()

def max_match(text):
    pos2 = len(text)
    result = ''
    while len(text) > 0:
        word = wordnet_lemmatizer.lemmatize(text[0:pos2])
        if word in wordlist:
            result = result + text[0:pos2] + ' '
            text = text[pos2:]
            pos2 = len(text)
        else:
            pos2 = pos2-1
    return result[0:-1]

if __name__ == '__main__':
    string = 'theyarebirds'
    print(max_match(string)) # they are birds
    '''
    当然，上述代码尚有一个不足，就是当字符串中存在非字母字符时（例如数字标点等），
    它可能会存在一些问题。有兴趣的读者不妨自己尝试完善改进这个版本的实现。
    '''
