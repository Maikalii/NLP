#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:28:05 2018

@author: zb
"""

import pandas as pd
import pickle
data = pd.read_csv('/Users/zb/Desktop/数据/东方头条网.csv')
import numpy as np
import os
import re
import jieba
import gc
output = "./output/"
from sklearn.model_selection import train_test_split


output = "./output/"

def saveObject(filename,obj):
    f=open(filename,'wb')
    pickle.dump(obj,f)
    f.close()
    return True

data_new = pd.DataFrame(data,columns= ['IR_ABSTRACT','CONTENT','MARK_LIST','NR_TYPE'])
train,test = train_test_split(data_new,test_size=0.2)

stopwords_path = '/Users/zb/Desktop/stopwords.txt'
f_stop = open(stopwords_path)
try:
     f_stop_text = f_stop.read( )
       # f_stop_text=unicode(f_stop_text,'utf-8')
finally:
    f_stop.close( )
f_stop_seg_list=f_stop_text.split('\n')
jieba.enable_parallel(4)

    
def jiebaclearText(text):
    mywordlist = []
    seg_list = jieba.cut(text, cut_all=False)
    liststr="/ ".join(seg_list)
   
    for myword in liststr.split('/'):
        if not(myword.strip() in f_stop_seg_list) and len(myword.strip())>1:
            mywordlist.append(myword)
    return mywordlist

train_set = []
docinfos = []

for index, row in train.iterrows():
    try:
        content = row['IR_ABSTRACT']
        word_list = jiebaclearText(content)
        train_set.append(word_list)
        detail={}
        detail["channel"]= row['MARK_LIST']
        detail["content"]= content
        docinfos.append(detail)
    except:
        pass

gc.collect()

from gensim import corpora,models,similarities,utils

#生成字典
dictionary = corpora.Dictionary(train_set)
#去除极低频的杂质词
dictionary.filter_extremes(no_below=1,no_above=1,keep_n=None)
#将词典保存下来，将语料也保存下来,语料转换成bow形式，方便后续使用
dictionary.save(output + "all.dic")
corpus = [dictionary.doc2bow(text) for text in train_set]
saveObject(output+"all.cps",corpus)
#存储原始的数据
saveObject(output+"all.info",docinfos)

#TF*IDF模型生成
#使用原始数据生成TFIDF模型
tfidfModel = models.TfidfModel(corpus)
#通过TFIDF模型生成TFIDF向量
tfidfVectors = tfidfModel[corpus]
#存储tfidfModel
tfidfModel.save(output + "allTFIDF.mdl")
indexTfidf = similarities.MatrixSimilarity(tfidfVectors)
indexTfidf.save(output + "allTFIDF.idx")


#LDA模型
lda = models.LdaModel(tfidfVectors, id2word=dictionary, num_topics=30)
lda.save(output + "allLDA50Topic.mdl")
corpus_lda = lda[tfidfVectors]
indexLDA = similarities.MatrixSimilarity(corpus_lda)
indexLDA.save(output + "allLDA50Topic.idx")

