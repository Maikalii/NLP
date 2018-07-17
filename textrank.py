#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 14:01:49 2018

@author: zb

本程序含有三个功能：
1.抽取出8类文章中个类别的前30个关键词，并生成表格
2.计算各类文章平均长度
3.计算各类文章平均独特词（*）的个数

2的结果处以3的结果可以得到一个比例，数值越大说明文章越冗杂，重复的词越多


独特词：distinct words，不重复的词的集合，例如文章‘苹果香
蕉梨苹果香蕉梨苹果香蕉梨苹果香蕉梨’文本长度为20但独特词为[苹果，香蕉，梨]。
"""

import jieba
import os,re,time,logging
import jieba.analyse
import pandas as pd
import pickle as pkl
import random
from textrank4zh import TextRank4Keyword



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
jieba.load_userdict("./user.txt")

class loadFolders(object):   # 迭代器
    def __init__(self,par_path):
        self.par_path = par_path
    def __iter__(self):
        for file in os.listdir(self.par_path):
            file_abspath = os.path.join(self.par_path, file)
            if os.path.isdir(file_abspath): # if file is a folder
                yield file_abspath
class loadFiles(object):
    def __init__(self,par_path):
        self.par_path = par_path
    def __iter__(self):
        folders = loadFolders(self.par_path)
        for folder in sorted(folders): # first level directory
            catg = folder.split(os.sep)[-1]
            #tmp = 0
            sample = random.sample(os.listdir(folder),500)
            for file in sample:
                 #secondary directory
                #if tmp > 899:
                #    break
                file_path = os.path.join(folder,file)
                if os.path.isfile(file_path):
                    this_file = open(file_path,'rb')
                    content = this_file.read().decode('utf8')
                    yield catg,content
                    this_file.close()
                #tmp +=1

if __name__=='__main__':
    path_doc_root = './train_data' # 根目录 即存放按类分类好文本集
    path_tmp = './data/textanalysis'
    path_keywordsbytextrank = os.path.join(path_tmp, 'keywordsbytextrank.pkl')
    path_keywordsbytfidf = os.path.join(path_tmp, 'keywordsbytfidf.pkl')
    path_keywordsbytextrank2f = os.path.join(path_tmp, 'keywordsbytextrank2f.pkl')
    path_average = os.path.join(path_tmp,'average.pkl')
    path_uniquewords = os.path.join(path_tmp,'uniquewords.pkl')
    path_ratio = os.path.join(path_tmp,'ratio.pkl')
    files = loadFiles(path_doc_root)
    l = ['']*8
    n = 1
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)
    for i,msg in enumerate(files):
        if i%n==0:
            catg = msg[0]
            file = msg[1]
            l[int(catg)] += file 
            #file = convert_doc_to_wordlist(file,cut_all=False)
            if int((i+1)/n) % 1000 == 0:
                print('{t} *** {i} \t docs has been dealed'
                      .format(i=i+1, t=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
    if not os.path.exists(path_keywordsbytextrank) or not os.path.exists(path_keywordsbytfidf) or not os.path.exists(path_keywordsbytextrank2f):    
        d1 = dict()
        d2 = dict()
        d3 = dict()
        jieba.analyse.set_stop_words('stopwords_new.txt')
        tittle_list = ['超热门','热门','高度关注','持续关注','开始关注','提神','极少兴趣','无人问津']
        for i,item in enumerate(l):
            words = jieba.analyse.textrank(item,topK=30)
            jieba.analyse.set_stop_words('stopwords_new.txt')
            words2 = jieba.analyse.extract_tags(item,topK = 30,
                                             allowPOS=('an','Ng','n'
                                                       ,'nr','ns','nt',
                                                       'nz','vg','v','vn','z','un' ))
            words3 = TextRank4Keyword(stop_words_file='stopwords_new.txt')
            words3.analyze(item,window = 5 , lower = True)
            w_list = words3.get_keywords(num = 30,word_min_len = 2)
            #d[str(i)] = ' '.join([word for word in words])
            d1[tittle_list[i]] = list(words)
            d2[tittle_list[i]] = list(words2)
            d3[tittle_list[i]] = list(w_list)
        df1 = pd.DataFrame.from_dict(d1)
        df2 = pd.DataFrame.from_dict(d2)
        df3 = pd.DataFrame.from_dict(d3)
        df1.to_pickle(path_keywordsbytextrank)
        df2.to_pickle(path_keywordsbytfidf)
        df3.to_pickle(path_keywordsbytextrank2f)
        del d1,d2,d3

    else:
        print('====Already got the keywords====')
        df1 = pd.read_pickle(path_keywordsbytextrank)
        df2 = pd.read_pickle(path_keywordsbytfidf)
        df3 = pd.read_pickle(path_keywordsbytextrank2f)
    if not os.path.exists(path_average):
        average = dict()
        for i,item in enumerate(l):
            average[tittle_list[i]] = len(item)/500
        x = open(path_average,'wb')
        pkl.dump(average,x)
        x.close
    else:
        print('====Already cacualated the average length====')
        with open(path_average,'rb') as f:
            average = pkl.load(f)
    if not os.path.exists(path_uniquewords) or not os.path.exists(path_ratio) :
        unique = dict()
        ratio = dict()
        for i,item in enumerate(l):
           words = list(jieba.cut(item,cut_all=False))
           uniwords = set(words) 
           ratio[tittle_list[i]] = len(words)/len(uniwords)
           unique[tittle_list[i]] = len(uniwords)/500
        x = open(path_uniquewords,'wb')
        pkl.dump(unique,x)
        x.close
        x = open(path_ratio,'wb')
        pkl.dump(ratio,x)
        x.close
    else:
        print('====Already got the uniquewords====')
        with open(path_uniquewords,'rb') as f:
            unique = pkl.load(f)
        with open(path_ratio,'rb') as f:
            ratio = pkl.load(f)