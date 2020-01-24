'''
TFIDF_review_model(query, csv_path)：
queryにクエリを，csv_pathに使用するデータセットのcsvを渡すと，クエリにマッチしている映画のリストを返す．レビュー版

TFIDF_synopsis_model(query, csv_path)：
queryにクエリを，csv_pathに使用するデータセットのcsvを渡すと，クエリにマッチしている映画のリストを返す．あらすじ版
'''

import pandas as pd
import os
import sys
import MeCab
import csv
import math

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# 浮動小数点を表示するときは小数点以下 2 桁で揃える
pd.options.display.float_format = '{:0.2f}'.format

#基本形で分かち書きをする関数
def analyze(text):
    
    tagger = MeCab.Tagger("-Ochasen")
    tagger.parse('')
    if type(text) is float:
        return []
    node = tagger.parseToNode(text)
    wakati = []
    while node:
        word = node.surface
        wclass = node.feature.split(',')
        #mecabは以下の順でnode.featureに情報を入れる　[0]品詞,[1]品詞細分類1,[2]品詞細分類2,[3]品詞細分類3,[4]活用形,[5]活用型,[6]原形,[7]読み,[8]発音
        if (
            #wclass[0] == '記号' or
            (wclass[0] == '形容詞' 
             and wclass[1] == '自立')or
            #wclass[0] == '助詞' or
            #wclass[0] == '助動詞' or
            #wclass[0] == '接続詞' or
            #wclass[0] == '接頭詞' or
            (wclass[0] == '動詞' 
             and (wclass[1] == '自立'))or
            #wclass[0] == '副詞' or 
            (wclass[0] == '名詞' and
             (wclass[1] == '一般' or 
              wclass[1] == '固有名詞' or
              wclass[1] == 'サ変接続' or
              wclass[1] == 'ナイ形容詞語幹' or
              wclass[1] == '形容動詞語幹' or
              wclass[1] == '副詞可能'))
            #wclass[0] == '連体詞'
        ):
            if wclass[6] != '*':
                wakati.append(wclass[6])
                
        node = node.next
    return wakati


def word_count(corpus):
    # 単語の数をカウントする
    count_vectorizer = CountVectorizer()
    X_count = count_vectorizer.fit_transform(corpus)

    # 見やすさのために表示するときは pandas のデータフレームにする
    df = pd.DataFrame(data=X_count.toarray(),
                      columns=count_vectorizer.get_feature_names())
    return X_count.toarray()
    
def word_num(corpus):
    # 単語の数をカウントする
    count_vectorizer = CountVectorizer()
    X_count = count_vectorizer.fit_transform(corpus)

    return len(count_vectorizer.get_feature_names())

def tf_idf(corpus):
    # scikit-learn の TF-IDF 実装
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(corpus)

    # TF-IDF を表示する
    df = pd.DataFrame(data=X_tfidf.toarray(),
                      columns=tfidf_vectorizer.get_feature_names())
    return df

def idf(corpus):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit_transform(corpus)
    feature = tfidf_vectorizer.get_feature_names()
    idf = tfidf_vectorizer._tfidf.idf_
    
    word_idf_dict = {}
    for pair in zip(feature, idf):
        word_idf_dict[pair[0]] = pair[1]
        
    return word_idf_dict

def tf(text):
    wakati = analyze(text)
    N = len(wakati)
    word_tf_dict = {}
    for word in wakati:
        tf = wakati.count(word)/N
        word_tf_dict[word] = tf
    return word_tf_dict

def make_tfidf_query(text, idf_dic):
    tf_dic = tf(text)
    tfidf_dic = idf_dic
    for key in tfidf_dic.keys():
        try:
            tfidf_dic[key] = tfidf_dic[key] * tf_dic[key]
        except KeyError:
            tfidf_dic[key] = 0.0
    
    return tfidf_dic

def comparison(tfidf_df, text, idf_dic):
    cos_sim = []
    tfidf_query = make_tfidf_query(text,idf_dic)
    for index in tfidf_df.index:
        cos_sim.append(make_cos(tfidf_df.loc[index], tfidf_query))
    return cos_sim
    
def make_cos(tfidf_df,tfidf_query):
    dot_product = 0
    tfidf_df = tfidf_df.to_dict()
    for key in tfidf_df.keys():
        dot_product += tfidf_df[key] * tfidf_query[key]
    
    len_df = 0
    len_query = 0
    
    for key in tfidf_df.keys():
        len_df += tfidf_df[key]*tfidf_df[key]
    len_df = math.sqrt(len_df)
    
    for key in tfidf_query.keys():
        len_query += tfidf_query[key]*tfidf_query[key]
    len_query = math.sqrt(len_query)
    
    if len_df*len_query == 0:
        return 0.0
    else:
        return dot_product/(len_df*len_query)

def TFIDF_review_model(query, csv_path):
    review_df = pd.read_csv(csv_path)    
    tfidf_df = tf_idf(review_df['reviews'])
	return comparison(tfidf_df, query, idf(review_df['reviews']))
	

def TFIDF_synopsis_model(query, csv_path):
	review_df = pd.read_csv(csv_path)    
    tfidf_df = tf_idf(review_df['synopsis'])
	return comparison(tfidf_df, query, idf(review_df['synopsis']))