#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 16:30:09 2018

@author: zb
"""
import gensim
from gensim.models import word2vec
import logging
import jieba
import math
from string import punctuation
from heapq import nlargest
from itertools import product, count
from gensim.models import word2vec
import numpy as np

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
jieba.load_userdict("./user.txt")
model = gensim.models.KeyedVectors.load_word2vec_format('./sgns.merge.word.bz2')
#momdel = word2vec.Word2Vec.load_word2vec_format('./sgns.merge.word.bz2',binary = True)
np.seterr(all='warn')

def cut_sentences(sentence):
    puns = frozenset(u'。！？')
    tmp = []
    for ch in sentence:
        tmp.append(ch)
        if puns.__contains__(ch):
            yield ''.join(tmp)
            tmp = []
    yield ''.join(tmp)

def create_stopwords():
    stop_list = [line.strip() for line in open("stopwords.txt", 'r', encoding='utf-8').readlines()]
    return stop_list

def two_sentences_similarity(sents_1, sents_2):
    '''
    计算两个句子的相似性
    :param sents_1:
    :param sents_2:
    :return:
    '''
    counter = 0
    for sent in sents_1:
        if sent in sents_2:
            counter += 1
    return counter / (math.log(len(sents_1) + len(sents_2)))

def create_graph(word_sent):
    """
    传入句子链表  返回句子之间相似度的图
    :param word_sent:
    :return:
    """
    num = len(word_sent)
    board = [[0.0 for _ in range(num)] for _ in range(num)]
 
    for i, j in product(range(num), repeat=2):
        if i != j:
            board[i][j] = compute_similarity_by_avg(word_sent[i], word_sent[j])
    return board

def cosine_similarity(vec1, vec2):
    '''
    计算两个向量之间的余弦相似度
    :param vec1:
    :param vec2:
    :return:
    '''
    tx = np.array(vec1)
    ty = np.array(vec2)
    cos1 = np.sum(tx * ty)
    cos21 = np.sqrt(sum(tx ** 2))
    cos22 = np.sqrt(sum(ty ** 2))
    cosine_value = cos1 / float(cos21 * cos22)
    return cosine_value

def compute_similarity_by_avg(sents_1, sents_2):
    '''
    对两个句子求平均词向量
    :param sents_1:
    :param sents_2:
    :return:
    '''
    if len(sents_1) == 0 or len(sents_2) == 0:
        return 0.0
    vec1 = model[sents_1[0]]
    for word1 in sents_1[1:]:
        vec1 = vec1 + model[word1]
 
    vec2 = model[sents_2[0]]
    for word2 in sents_2[1:]:
        vec2 = vec2 + model[word2]
 
    similarity = cosine_similarity(vec1 / len(sents_1), vec2 / len(sents_2))
    return similarity

def calculate_score(weight_graph, scores, i):
    """
    计算句子在图中的分数
    :param weight_graph:
    :param scores:
    :param i:
    :return:
    """
    length = len(weight_graph)
    d = 0.85
    added_score = 0.0
 
    for j in range(length):
        fraction = 0.0
        denominator = 0.0
        # 计算分子
        fraction = weight_graph[j][i] * scores[j]
        # 计算分母
        for k in range(length):
            denominator += weight_graph[j][k]
            if denominator == 0:
                denominator = 1
        added_score += fraction / denominator
    # 算出最终的分数
    weighted_score = (1 - d) + d * added_score
    return weighted_score
 
 
def weight_sentences_rank(weight_graph):
    '''
    输入相似度的图（矩阵)
    返回各个句子的分数
    :param weight_graph:
    :return:
    '''
    # 初始分数设置为0.5
    scores = [0.5 for _ in range(len(weight_graph))]
    old_scores = [0.0 for _ in range(len(weight_graph))]
 
    # 开始迭代
    while different(scores, old_scores):
        for i in range(len(weight_graph)):
            old_scores[i] = scores[i]
        for i in range(len(weight_graph)):
            scores[i] = calculate_score(weight_graph, scores, i)
    return scores
 
 
def different(scores, old_scores):
    '''
    判断前后分数有无变化
    :param scores:
    :param old_scores:
    :return:
    '''
    flag = False
    for i in range(len(scores)):
        if math.fabs(scores[i] - old_scores[i]) >= 0.0001:
            flag = True
            break
    return flag
 
 
def filter_symbols(sents):
    stopwords = create_stopwords() + ['。', ' ', '.','，','？','！']
    _sents = []
    for sentence in sents:
        for word in sentence:
            if word in stopwords:
                sentence.remove(word)
        if sentence:
            _sents.append(sentence)
    return _sents
 
 
def filter_model(sents):
    _sents = []
    for sentence in sents:
        for word in sentence:
            if word not in model:
                #print(word)
                sentence.remove(word)
        if sentence:
            _sents.append(sentence)
    return _sents
 
 

def summarize(text, n):
    '''
    text为输入文本
    n为想要返回句子的数量（通常为2-3）
    :param text:
    :param n:
    :return:
    '''
    tokens = cut_sentences(text)
    sentences = []
    sents = []
    for sent in tokens:
        sentences.append(sent)
        sents.append([word for word in jieba.cut(sent,cut_all=False) if word])
 
    sents = filter_symbols(sents)
    sents = filter_model(sents)
    sents = filter_model(sents)
    sents = filter_model(sents)
    #print(sents)
    graph = create_graph(sents)
 
    scores = weight_sentences_rank(graph)
    sent_selected = nlargest(n, zip(scores, count()))
    sent_index = []
    for i in range(n):
        sent_index.append(sent_selected[i][1])
    return [sentences[i] for i in sent_index]



if __name__ == '__main__':
    with open("./train_data/0/10463150.txt", "r", encoding='utf-8') as myfile:
        text = myfile.read().replace('\n', '')
        abs_sent = summarize(text, 3)
        print(''.join(abs_sent))

