#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:59:06 2018

@author: zb
"""

from gensim import corpora,models,similarities,utils
import jieba
import jieba.posseg as pseg
import sys
import os
import re
import gc
import pickle
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

output = "./output/"
def saveObject(filename,obj):
    f=open(filename,'wb')
    pickle.dump(obj,f)
    f.close()
    return True
def loadObject(filename):
    f=open(filename,'rb')
    obj=pickle.load(f)
    return obj


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

docinfos = loadObject(output + "all.info")
#载入字典
dictionary = corpora.Dictionary.load(output + "all.dic")
tfidfModel = models.TfidfModel.load(output+"allTFIDF.mdl")
indexTfidf = similarities.MatrixSimilarity.load(output + "allTFIDF.idx")

#载入LDA索引
ldaModel = models.LdaModel.load(output + "allLDA50Topic.mdl")
indexLDA = similarities.MatrixSimilarity.load(output + "allLDA50Topic.idx")

query= """
取舍之道，说起来容易，做起来却很难。有时候它意味着你要放弃自己擅长的、甚至给你带来成功的东西，而去在全新的、前景未知的领域做出尝试。如果柯达能早点从胶片中走出来、诺基亚能少造点塞班机，也许就不会有今天的佳能、苹果、三星。如果保时捷没有冒天下之大不韪造出卡宴，也很难预料它如今的境遇会如何。
看开头这应该是一篇为全新宝马X1 Li（下文简称新X1）洗地的文章，我想很多宝马死忠、车神也已经准备移步评论区，敲下神圣、激扬的文字。其实这年月已经不流行道德绑架了，所以对于车迷而言，也请适当的放下你对这个品牌的喜爱，试着从一个兜里揣着三十来万、想买辆豪华品牌SUV的消费者的角度来看待新X1，这也是本文作者的角度。
一切的争议其实都是由UKL前驱平台而起，宝马做了一个艰难的决定，给几个入门的车系换上了前驱平台。在空间、成本相对有限的入门车型里，放弃后驱传统以换取更大空间。我想这也能看出宝马对于未来趋势的判断，就是在难以兼顾空间、操控的入门豪华车市场，消费者会重视空间实用性多过操控性，宝马也将宝押在了空间实用性上。
宝马做出了取舍，消费者该怎么选？以我自己为例吧，在买车的时候也考虑过老款宝马X1（下文简称老X1），优惠后25万左右，价格没比途观、奇骏贵多少，但品牌、动力的提升都可谓巨大。但为什么最后放弃？正是因为空间。如果家里只有这一辆车，老X1的空间确实有些力不从心。
新X1的空间绝对没有问题，而且看起来也比老款要更大气、更有面子。我们测试的这台xDrive25Li 豪华型同样采用2.0T和8AT变速箱，只是改为基于前驱的四驱系统。究竟值不值得选择，读完文章希望你能有答案。
更大气、更阳刚新X1在外观上的变化翻天覆地，与之前的老X1是完全性格的两种产物。老X1低调内敛，有着一种含蓄之美；而新X1则阳刚帅气，骨子里透着一种坚韧的性格，有着更符合男性消费者需求的阳刚之美。
"""
query_bow = dictionary.doc2bow(filter(lambda x: len(x)>0,jiebaclearText(query)))
tfidfvect = tfidfModel[query_bow]
simstfidf = indexTfidf[tfidfvect]
sort_sims = sorted(enumerate(simstfidf), key=lambda item: -item[1])
print("TFIDF similary Top 10:::")
for sim in sort_sims[:10]:
    print("ID : " + docinfos[sim[0]]["channel"] + "\t" + docinfos[sim[0]]["content"] + "\tsimilary:::" + str(sim[1]))


ldavec = ldaModel[tfidfvect]
simlda = indexLDA[ldavec]
sort_sims = sorted(enumerate(simlda), key=lambda item: -item[1])
print("LDA similary Top 10:::")
for sim in sort_sims[:10]:
    print("ID : " + str(docinfos[sim[0]]["channel"]) + "\t" + docinfos[sim[0]]["content"] + "\tsimilary:::" + str(sim[1]))

