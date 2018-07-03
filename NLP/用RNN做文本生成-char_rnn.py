# -*- coding: utf-8 -*-
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

'''
用RNN做文本生成
举个小小的例子，来看看 LSTM 是怎么玩的
我们这里用温斯顿丘吉尔的人物传记作为我们的学习语料。
(各种中文语料可以自行网上查找， 英文的小说语料可以从古登堡计划网站下载txt平文本：
https://www.gutenberg.org/wiki/Category:Bookshelf
第一步，先导入各种库
'''

'''
Using Theano backend.
Using gpu device 0: Tesla K80 (CNMeM is disabled, cuDNN 5105)
/usr/local/lib/python3.5/dist-packages/theano/sandbox/cuda/__init__.py:600: UserWarning: Your cuDNN version is more recent than the one Theano officially supports. If you see any problems, try updating Theano or downgrading cuDNN to version 5.
  warnings.warn(warn)
接下来，我们把文本读入
'''
'''
open('input/74-0.txt','r') 读取文件时报错
"UnicodeDecodeError: 'gbk' codec can't decode byte 0x80 in position 205: illegal multibyte sequence"
解决办法1  open('order.log','r', encoding='UTF-8')
解决办法2  open('order.log','rb')
'''
raw_text = open(r'./input/Winston_Churchil.txt','r',encoding='utf-8').read()
raw_text = raw_text.lower()
'''
# print( raw_text ) #读文件 \r\n
反斜杠(捺)r 换行 相当于回车符
反斜杠(捺)n 新行 换行符
反斜杠(捺)a 警报
退格符 反斜杠(捺)b
换页符 反斜杠(捺)f
Tab制表符 反斜杠(捺)t
垂直 Tab 符 反斜杠(捺)v
使用数字指定的Unicode字符 反斜杠(捺)u，如 反斜杠(捺)u2000
使用十六进制数指定的Unicode字符 反斜杠(捺)x,如 反斜杠(捺)xc8
空值 反斜杠(捺)0 zero
'''

'''
既然我们是以每个字母为层级，字母总共才26个，
所以可很方便的用One-Hot来编码出所有的字母（当然，可能还有些标点符号和其他noise）
'''
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
print( chars , len(chars)) # 全部的chars
'''
['\n',
 ' ',
 '!',
 '#',
 '$',
 '%',
 '(',
 ')',
 '*',
 ',',
 '-',
 '.',
 '/',
 '0',
 '1',
 '2',
 '3',
 '4',
 '5',
 '6',
 '7',
 '8',
 '9',
 ':',
 ';',
 '?',
 '@',
 '[',
 ']',
 '_',
 'a',
 'b',
 'c',
 'd',
 'e',
 'f',
 'g',
 'h',
 'i',
 'j',
 'k',
 'l',
 'm',
 'n',
 'o',
 'p',
 'q',
 'r',
 's',
 't',
 'u',
 'v',
 'w',
 'x',
 'y',
 'z',
 '‘',
 '’',
 '“',
 '”',
 '\ufeff']
一共有：61 种字符
'''
print( len(raw_text) ) # 原文本字符总数: 276830
'''
这里简单的文本预测是，给出前置的字母，预测下一个字母。
比如，给出前置字母 Winsto 预测下一个字母是 n

构造训练测试集，把 raw_text 变成可用来训练的 x , y (x 是前置字母们 y 是后一个字母)
'''
seq_length = 100
x , y = [] , []
for i in range(0, len(raw_text) - seq_length):
    given = raw_text[i:i + seq_length]
    predict = raw_text[i + seq_length]
    x.append([char_to_int[char] for char in given])
    y.append(char_to_int[predict])

print(x[:3])
print(y[:3])
'''
[[60, 45, 47, 44, 39, 34, 32, 49, 1, 36, 50, 49, 34, 43, 31, 34, 47, 36, 57,
48, 1, 47, 34, 30, 41, 1, 48, 44, 41, 33, 38, 34, 47, 48, 1, 44, 35, 1, 35,
 44, 47, 49, 50, 43, 34, 9, 1, 31, 54, 1, .......]]
[35, 1, 30]

此刻，楼上这些表达方式，类似就是一个词袋，或者说 index

接下来我们做两件事：
1、把input的数字表达（index）变成 LSTM 需要的数组格式： [样本数，时间步伐，特征]
2、对于output，用one-hot编码做output的预测可以给我们更好的效果，相对于直接预测一个准确的y数值的话。
'''
n_patterns = len(x)
n_vocab = len(chars)

# 把x变成LSTM需要的样子
x = numpy.reshape(x, (n_patterns, seq_length, 1))
# 简单normal到0-1之间
x = x / float(n_vocab)
# output变成one-hot
y = np_utils.to_categorical(y)

print(x[11])
print(y[11])

'''
[[ 0.80327869]
 [ 0.55737705]
 [ 0.70491803]
 [ 0.50819672]
 [ 0.55737705]
 [ 0.7704918 ]
 [ 0.59016393]
 [ 0.93442623]
 [ 0.78688525]
 [ 0.01639344]
 [ 0.7704918 ]
 [ 0.55737705]
 [ 0.49180328]
 [ 0.67213115]
 [ 0.01639344]
 [ 0.78688525]
 [ 0.72131148]
 [ 0.67213115]
 [ 0.54098361]
 [ 0.62295082]
 [ 0.55737705]
 [ 0.7704918 ]
 [ 0.78688525]
 [ 0.01639344]
 [ 0.72131148]
 [ 0.57377049]
 [ 0.01639344]
 [ 0.57377049]
 [ 0.72131148]
 [ 0.7704918 ]
 [ 0.80327869]
 [ 0.81967213]
 [ 0.70491803]
 [ 0.55737705]
 [ 0.14754098]
 [ 0.01639344]
 [ 0.50819672]
 [ 0.8852459 ]
 [ 0.01639344]
 [ 0.7704918 ]
 [ 0.62295082]
 [ 0.52459016]
 [ 0.60655738]
 [ 0.49180328]
 [ 0.7704918 ]
 [ 0.54098361]
 [ 0.01639344]
 [ 0.60655738]
 [ 0.49180328]
 [ 0.7704918 ]
 [ 0.54098361]
 [ 0.62295082]
 [ 0.70491803]
 [ 0.59016393]
 [ 0.01639344]
 [ 0.54098361]
 [ 0.49180328]
 [ 0.83606557]
 [ 0.62295082]
 [ 0.78688525]
 [ 0.        ]
 [ 0.        ]
 [ 0.80327869]
 [ 0.60655738]
 [ 0.62295082]
 [ 0.78688525]
 [ 0.01639344]
 [ 0.55737705]
 [ 0.50819672]
 [ 0.72131148]
 [ 0.72131148]
 [ 0.6557377 ]
 [ 0.01639344]
 [ 0.62295082]
 [ 0.78688525]
 [ 0.01639344]
 [ 0.57377049]
 [ 0.72131148]
 [ 0.7704918 ]
 [ 0.01639344]
 [ 0.80327869]
 [ 0.60655738]
 [ 0.55737705]
 [ 0.01639344]
 [ 0.81967213]
 [ 0.78688525]
 [ 0.55737705]
 [ 0.01639344]
 [ 0.72131148]
 [ 0.57377049]
 [ 0.01639344]
 [ 0.49180328]
 [ 0.70491803]
 [ 0.8852459 ]
 [ 0.72131148]
 [ 0.70491803]
 [ 0.55737705]
 [ 0.01639344]
 [ 0.49180328]
 [ 0.70491803]]
[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  1.  0.  0.  0.  0.  0.]
模型建造
LSTM模型构建
'''
model = Sequential()
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

'''
跑模型
'''
model.fit(x, y, nb_epoch=50, batch_size=4096)
'''
Epoch 1/50
276730/276730 [==============================] - 197s - loss: 3.1120
Epoch 2/50
276730/276730 [==============================] - 197s - loss: 3.0227
Epoch 3/50
276730/276730 [==============================] - 197s - loss: 2.9910
Epoch 4/50
276730/276730 [==============================] - 197s - loss: 2.9337
Epoch 5/50
276730/276730 [==============================] - 197s - loss: 2.8971
Epoch 6/50
276730/276730 [==============================] - 197s - loss: 2.8784
Epoch 7/50
276730/276730 [==============================] - 197s - loss: 2.8640
Epoch 8/50
276730/276730 [==============================] - 197s - loss: 2.8516
Epoch 9/50
276730/276730 [==============================] - 197s - loss: 2.8384
Epoch 10/50
276730/276730 [==============================] - 197s - loss: 2.8254
Epoch 11/50
276730/276730 [==============================] - 197s - loss: 2.8133
Epoch 12/50
276730/276730 [==============================] - 197s - loss: 2.8032
Epoch 13/50
276730/276730 [==============================] - 197s - loss: 2.7913
Epoch 14/50
276730/276730 [==============================] - 197s - loss: 2.7831
Epoch 15/50
276730/276730 [==============================] - 197s - loss: 2.7744
Epoch 16/50
276730/276730 [==============================] - 197s - loss: 2.7672
Epoch 17/50
276730/276730 [==============================] - 197s - loss: 2.7601
Epoch 18/50
276730/276730 [==============================] - 197s - loss: 2.7540
Epoch 19/50
276730/276730 [==============================] - 197s - loss: 2.7477
Epoch 20/50
276730/276730 [==============================] - 197s - loss: 2.7418
Epoch 21/50
276730/276730 [==============================] - 197s - loss: 2.7360
Epoch 22/50
276730/276730 [==============================] - 197s - loss: 2.7296
Epoch 23/50
276730/276730 [==============================] - 197s - loss: 2.7238
Epoch 24/50
276730/276730 [==============================] - 197s - loss: 2.7180
Epoch 25/50
276730/276730 [==============================] - 197s - loss: 2.7113
Epoch 26/50
276730/276730 [==============================] - 197s - loss: 2.7055
Epoch 27/50
276730/276730 [==============================] - 197s - loss: 2.7000
Epoch 28/50
276730/276730 [==============================] - 197s - loss: 2.6934
Epoch 29/50
276730/276730 [==============================] - 197s - loss: 2.6859
Epoch 30/50
276730/276730 [==============================] - 197s - loss: 2.6800
Epoch 31/50
276730/276730 [==============================] - 197s - loss: 2.6741
Epoch 32/50
276730/276730 [==============================] - 197s - loss: 2.6669
Epoch 33/50
276730/276730 [==============================] - 197s - loss: 2.6593
Epoch 34/50
276730/276730 [==============================] - 197s - loss: 2.6529
Epoch 35/50
276730/276730 [==============================] - 197s - loss: 2.6461
Epoch 36/50
276730/276730 [==============================] - 197s - loss: 2.6385
Epoch 37/50
276730/276730 [==============================] - 197s - loss: 2.6320
Epoch 38/50
276730/276730 [==============================] - 197s - loss: 2.6249
Epoch 39/50
276730/276730 [==============================] - 197s - loss: 2.6187
Epoch 40/50
276730/276730 [==============================] - 197s - loss: 2.6110
Epoch 41/50
276730/276730 [==============================] - 192s - loss: 2.6039
Epoch 42/50
276730/276730 [==============================] - 141s - loss: 2.5969
Epoch 43/50
276730/276730 [==============================] - 140s - loss: 2.5909
Epoch 44/50
276730/276730 [==============================] - 140s - loss: 2.5843
Epoch 45/50
276730/276730 [==============================] - 140s - loss: 2.5763
Epoch 46/50
276730/276730 [==============================] - 140s - loss: 2.5697
Epoch 47/50
276730/276730 [==============================] - 141s - loss: 2.5635
Epoch 48/50
276730/276730 [==============================] - 140s - loss: 2.5575
Epoch 49/50
276730/276730 [==============================] - 140s - loss: 2.5496
Epoch 50/50
276730/276730 [==============================] - 140s - loss: 2.5451
Out[11]:
<keras.callbacks.History at 0x7fb6121b6e48>
我们来写个程序，看看我们训练出来的LSTM的效果：
'''
def predict_next(input_array):
    x = numpy.reshape(input_array, (1, seq_length, 1))
    x = x / float(n_vocab)
    y = model.predict(x)
    return y

def string_to_index(raw_input):
    res = []
    for c in raw_input[(len(raw_input)-seq_length):]:
        res.append(char_to_int[c])
    return res

def y_to_char(y):
    largest_index = y.argmax()
    c = int_to_char[largest_index]
    return c

'''
好，写成一个大程序：
'''
def generate_article(init, rounds=200):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_char(predict_next(string_to_index(in_string)))
        in_string += n
    return in_string

init = 'His object in coming to New York was to engage officers for that service. He came at an opportune moment'
article = generate_article(init)
print(article)

'''
his object in coming to new york was to engage officers for that service.
he came at an opportune moment th the toote of the carie
and the soote of the carie and the soote of the carie and the soote of the carie
and the soote of the carie and the soote of the carie and the soote of the carie and the soo
'''