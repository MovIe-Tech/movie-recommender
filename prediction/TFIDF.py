import pandas as pd
import os
import sys
import MeCab
import csv
import math

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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

def tf(query_list):
    wakati = query_list
    N = len(wakati)
    word_tf_dict = {}
    for word in wakati:
        tf = wakati.count(word)/N
        word_tf_dict[word] = tf
    return word_tf_dict

def make_tfidf_query(query_list, idf_dic):
    tf_dic = tf(query_list)
    tfidf_dic = idf_dic
    for key in tfidf_dic.keys():
        try:
            tfidf_dic[key] = tfidf_dic[key] * tf_dic[key]
        except KeyError:
            tfidf_dic[key] = 0.0
    
    return tfidf_dic

def comparison(tfidf_df, query_list, idf_dic):
    cos_sim = []
    tfidf_query = make_tfidf_query(query_list, idf_dic)
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

def TFIDF_pred_review(query, csv_path):
	review_df = pd.read_csv(csv_path)
	tfidf_df = tf_idf(review_df['reviews'])
	return cos_list=comparison(tfidf_df, query, idf(review_df['reviews']))
	
def TFIDF_pred_synopsis(query, csv_path):
	review_df = pd.read_csv(csv_path)
	tfidf_df = tf_idf(review_df['synopsis'])
	return cos_list=comparison(tfidf_df, query, idf(review_df['synopsis']))