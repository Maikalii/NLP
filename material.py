#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:07:22 2018

@author: zb
"""

import pandas as pd
data = pd.read_csv('/Users/zb/Desktop/数据/东方头条网.csv')
import numpy as np
from sklearn.model_selection import train_test_split

train,test = train_test_split(data_new,test_size=0.2)
s = ''
for i in train['IR_ABSTRACT']:
    if type(i) == str:
        s = s +  '\n' + i
f = open("train.txt",'w+')
f.write(s)
f.close()

from os import path
import jieba.analyse as analyse

text_path = '/Users/zb/Desktop/text.txt' #设置要分析的文本路径
text = open(text_path).read()
data = text.encode('utf8')

text1 = jiebaclearText(text)

train_set = []
for line in train['IR_ABSTRACT']:
    try:
        word_list = jiebaclearText(line)
        train_set.append(word_list)
    except:
        pass



from gensim import corpora,models,similarities,utils
dictionary = corpora.Dictionary(train_set)
#去除极低频的杂质词
dictionary.filter_extremes(no_below=1,no_above=1,keep_n=None)
#将词典保存下来，方便后续使用
dictionary.save(output + "all.dic")