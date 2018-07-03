import nltk
from nltk.corpus import brown
help(nltk.corpus.brown)
'''
# 在 terminal 里输入 nltk.download() 即可下载相应的所需包
Natural Language Toolkit 自然语言处理工具包，在NLP领域中，最常使用的一个Python库。
NLTK是一个开源的项目，包含:Python模块，数据集和教程，用于NLP的研究和开发。
NLTK由Steven Bird和Edward Loper在宾夕法尼亚大学计算机和信息科学系开发。
NLTK包括图形演示和示例数据。其提供的教程解释了工具包支持的语言处理任务背后的基本概念。
'''
'''
HMM实例-进行词性标注【用NLTK自带的Brown词库进行学习】
单词集 words = w1 ... wN
Tag集 tags = t1 ... tN
P(tags | words) 正比于 P(ti | t{i-1}) * P(wi | ti)

为了找一个句子的tag，其实就是找最好的一套tags，让它最能够符合给定的单词(words)。
'''
'''
（1）预处理词库 【做预处理，即给 words 加上开始和结束符号】
Brown 里的句子都是已标注好的( 单词 , 词性 )，词性包括：NOUN 名词、VERB 动词 等。
长这个样子 (I , NOUN), (LOVE, VERB), (YOU, NOUN) # I 名词
那么，我们的开始符号也得跟他的格式符合，用 (START, START) (END, END) 来表示
'''
brown_tags_words = []
for sent in brown.tagged_sents():
    brown_tags_words.append(("START", "START")) # 先加开头
    # 把tag都省略成前两个字母 tag[:2]
    brown_tags_words.extend([(tag[:2], word) for (word, tag) in sent])
    brown_tags_words.append(("END", "END")) # 加个结尾

'''
（2）词统计，将所有的词库中的 word单词 与 tag 之间的关系，做个简单粗暴的统计。
也就是之前提到的：P(wi | ti) = count(wi, ti) / count(ti)
你可以一个个的 loop 全部的 corpus 语料库，
NLTK自带统计工具（没有什么必要hack，装起逼来也不X，想自己实现，可以去实现，不想的话，就用这里提供的方法）
nltk.ConditionalFreqDist 条件频率分布 conditional frequency distribution
nltk.ConditionalProbDist 条件概率分布 conditional probability distribution
'''
cfd_tagwords = nltk.ConditionalFreqDist(brown_tags_words)
cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)
print("The probability of an adjective (JJ) being 'new' is", cpd_tagwords["JJ"].prob("new"))
# 形容词（JJ）为“new”的概率是      prob 概率的简写
print("The probability of a verb (VB) being 'duck' is", cpd_tagwords["VB"].prob("duck"))
'''
（3）计算公式：P(ti | t{i-1}) = count(t{i-1}, ti) / count(t{i-1})
这个公式跟words没有什么卵关系。它是属于隐层的马科夫链。
nltk.bigrams二元随机存储器，将前后两个一组，联在一起
'''
brown_tags = [tag for (tag, word) in brown_tags_words]  #获取所有tag
cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(brown_tags)) # count(t{i-1} , ti)
cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist) # P(ti | t{i-1})
print("If we have just seen 'DT', the probability of 'NN' is", cpd_tags["DT"].prob("NN"))
print("If we have just seen 'VB', the probability of 'JJ' is", cpd_tags["VB"].prob("DT"))
print("If we have just seen 'VB', the probability of 'NN' is", cpd_tags["VB"].prob("NN"))
'''
一些有趣的结果：比如， 一句话 "I want to race"， 一套tag "PP VB TO VB"
他们之间的匹配度有多高呢？
其实就是：P(START) * P(PP|START) * P(I | PP) * P(VB | PP) * P(want | VB) *
        P(TO | VB) * P(to | TO) * P(VB | TO) * P(race | VB) * P(END | VB)
'''
prob_tagsequence = cpd_tags["START"].prob("PP") * cpd_tagwords["PP"].prob("I") * \
                   cpd_tags["PP"].prob("VB") * cpd_tagwords["VB"].prob("want") * \
                   cpd_tags["VB"].prob("TO") * cpd_tagwords["TO"].prob("to") * \
                   cpd_tags["TO"].prob("VB") * cpd_tagwords["VB"].prob("race") * \
                   cpd_tags["VB"].prob("END")

print("The probability of the tag sequence 'START PP VB TO VB END' "
      "for 'I want to race' is:", prob_tagsequence)

'''
（4）维特比 Viterbi 的实现  --  如果有一句话，怎么计算最符合的 tag 是哪组呢？
首先，拿出所有独特的 tags（也就是tags的全集）
'''
distinct_tags = set(brown_tags)  # distinct 不同的
sentence = ["I", "want", "to", "race"] # 我想参加比赛
sentlen = len(sentence)
'''
接下来，开始维特比：从 1 循环到句子的总长N，记为i。每次都找出以 tag X 为最终节点，长度为i的tag链。
'''
viterbi = []
'''
同时，还需要一个回溯器：从1循环到句子的总长N，记为i。把所有tag X 前一个Tag记下来。
'''
backpointer = []
first_viterbi = {}
first_backpointer = {}
for tag in distinct_tags:
    if tag == "START": continue # don't record anything for the START tag
    first_viterbi[tag] = cpd_tags["START"].prob(tag) * cpd_tagwords[tag].prob(sentence[0])
    first_backpointer[tag] = "START"

print(first_viterbi)
print(first_backpointer)

viterbi.append(first_viterbi)
backpointer.append(first_backpointer)

currbest = max(first_viterbi.keys(), key=lambda tag: first_viterbi[tag])
print("Word", "'" + sentence[0] + "'", "current best two-tag sequence:",
      first_backpointer[currbest], currbest)

for wordindex in range(1, len(sentence)):
    this_viterbi = {}
    this_backpointer = {}
    prev_viterbi = viterbi[-1]

    for tag in distinct_tags:
        if tag == "START": # START没啥卵用，要忽略
            continue
        '''
        如果现在这个tag是X，现在的单词是w，
        想找前一个tag Y，且让最好的 tag sequence 以 Y X 结尾。
        也就是说，Y要能最大化：prev_viterbi[ Y ] * P(X | Y) * P( w | X)
        '''
        best_previous = max(prev_viterbi.keys(),
                            key=lambda prevtag: prev_viterbi[prevtag] \
                                                * cpd_tags[prevtag].prob(tag) \
                                                * cpd_tagwords[tag].prob(sentence[wordindex]))

        this_viterbi[tag] = prev_viterbi[best_previous] \
                            * cpd_tags[best_previous].prob(tag) \
                            * cpd_tagwords[tag].prob(sentence[wordindex])
        this_backpointer[tag] = best_previous

    # 每次找完Y 都要把目前最好的 存一下
    currbest = max(this_viterbi.keys(), key=lambda tag: this_viterbi[tag])
    print("Word", "'" + sentence[wordindex] + "'", "current best two-tag sequence:"
          , this_backpointer[currbest], currbest)

    # 全部存下来
    viterbi.append(this_viterbi)
    backpointer.append(this_backpointer)


# 找所有以END结尾的tag sequence
prev_viterbi = viterbi[-1]
best_previous = max(prev_viterbi.keys(),
                    key=lambda prevtag: prev_viterbi[prevtag] * cpd_tags[prevtag].prob("END"))

prob_tagsequence = prev_viterbi[best_previous] * cpd_tags[best_previous].prob("END")

# 倒着存。。。。因为。。好的在后面
best_tagsequence = ["END", best_previous]
backpointer.reverse()

current_best_tag = best_previous
for bp in backpointer:
    best_tagsequence.append(bp[current_best_tag])
    current_best_tag = bp[current_best_tag]

best_tagsequence.reverse()

print("The sentence was:", end=" ") # end=" " print 输出不换行
for w in sentence: print(w, end=" ")
print("\nThe best tag sequence is:", end=" ")
for t in best_tagsequence: print(t, end=" ")
print("\nThe probability of the best tag sequence is:", prob_tagsequence)
# 最佳标签序列的概率为 5.71772824864617e-14