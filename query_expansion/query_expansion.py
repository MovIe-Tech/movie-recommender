"""
クエリの類義語を辞書から抽出し，クエリが表す意味の範囲を拡張す
"""

import pandas as pd
import sqlite3
import string
from random import randint
from pprint import pprint
from natto import MeCab
import numpy as np

def txt2words(txt) -> list:
    posid = [36,37,38,40,41,42,43,44,45,46,47,50,51,52,66,67,2,31,36,10,34]
    words = []
    parser = MeCab()
    nodes = parser.parse(txt,as_nodes=True)
    for node in nodes:
        if not node.is_eos():
            feature = node.feature.split(',')
            if node.posid in posid and feature[6] != "*":
                words.append(feature[6]) 
    return words

def clean(word_list) -> list:
    stws = pd.read_csv("./data/stopwords.csv", encoding='utf-8').T.values.tolist()[0]
    return [word for word in word_list if word not in stws]

conn = sqlite3.connect("/Users/zmn/Documents/mit/movie-recommender/query_expansion/wnjpn.db")

# 英語文字があるかを判断する関数
def __isLetter(word) -> bool:
    s = set(string.ascii_lowercase)
    return (word[0] in s)

# 特定の単語を入力とした時に、類義語を検索する関数
def __search_similar_words(word) -> list:
    similar_words = []
    cur = conn.execute("select wordid from word where lemma='%s'" % word)
    word_id = 99999999  #temp 
    for row in cur:
        word_id = row[0]
    if word_id==99999999:
        return []
    cur = conn.execute("select synset from sense where wordid='%s'" % word_id)
    synsets = []
    for row in cur:
        synsets.append(row[0])

    for synset in synsets:
        cur = conn.execute("select wordid from sense where (synset='%s' and wordid!=%s)" % (synset,word_id))
        for row in cur:
             
            target_word_id = row[0]
            cur_1 = conn.execute("select lemma from word where wordid=%s" % target_word_id)
            for row_1 in cur_1:
                if not __isLetter(row_1[0]):
                    similar_words.append(row_1[0])
    return similar_words



def expansion_magic(query, rate) -> list:
    """This function expanses a query of string type to a list of sentenses

        param query: a query of string type
        param rate: rate determines how many ways an query can be expressed
        return: a list of a rate quantity of queries words 
    """
    
    word_list = clean(txt2words(query))
    querys = [word_list]

    synsets = []
    for w in word_list:
        synsets.append(__search_similar_words(w))

    for _ in range(rate):
        similar_word_list = []
        for index, synset in enumerate(synsets):
            if synset == []:
                similar_word_list.append(word_list[index])
            else:
                random_index = randint(0, len(synset)-1)
                similar_word_list.append(synset[random_index])
                synset.remove(synset[random_index])
        querys.append(similar_word_list)
    
    return querys

pprint(expansion_magic("母は泣いた", 20))
