# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import jieba
import sys
stopwords_path = '/Users/zb/Desktop/stopwords.txt'

def jiebaclearText(text):
    mywordlist = []
    seg_list = jieba.cut(text, cut_all=False)
    liststr="/ ".join(seg_list)
    f_stop = open(stopwords_path)
    try:
        f_stop_text = f_stop.read( )
       # f_stop_text=unicode(f_stop_text,'utf-8')
    finally:
        f_stop.close( )
    f_stop_seg_list=f_stop_text.split('\n')
    for myword in liststr.split('/'):
        if not(myword.strip() in f_stop_seg_list) and len(myword.strip())>1:
            mywordlist.append(myword)
    return ''.join(mywordlist)


def clean(s):
    s = s.replace('\u3000','')
    s = s.replace('\r','')
    s = s.replace('\n','')
    s = s.replace('\t','')
    return s