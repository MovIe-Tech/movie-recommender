import pandas as pd
import os
import sys
import MeCab

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#出現の割合の下限値を入力すると，それに満たない語を出力
def stop_word_min(corpus, num):
    count_vectorizer = CountVectorizer(max_df=num)
    X_count = count_vectorizer.fit_transform(corpus)
    
    return count_vectorizer.get_feature_names()
    
#出現の割合の上限値を入力すると，それを超える語を出力
def stop_word_max(corpus, num):
    count_vectorizer = CountVectorizer(min_df=num)
    X_count = count_vectorizer.fit_transform(corpus)
    
    return count_vectorizer.get_feature_names()
    
review_df = pd.read_csv("1005data.csv")
review_df = review_df.rename(columns={'Unnamed: 0':'index', 'title':'title', 'rate':'score', 'reviews':'review'}) 

#文書を分かち書きにする関数
def analyze(text):
    tagger = MeCab.Tagger("-Ochasen")
    tagger.parse('')
    node = tagger.parseToNode(text)
    wakati = ''
    while node:
        word = node.surface
        wclass = node.feature.split(',')
        #mecabは以下の順でnode.featureに情報を入れる　[0]品詞,[1]品詞細分類1,[2]品詞細分類2,[3]品詞細分類3,[4]活用形,[5]活用型,[6]原形,[7]読み,[8]発音
        if (
            #wclass[0] == '記号' or
            wclass[0] == '形容詞' or
            #wclass[0] == '助詞' or
            #wclass[0] == '助動詞' or
            #wclass[0] == '接続詞' or
            #wclass[0] == '接頭詞' or
            wclass[0] == '動詞' or
            #wclass[0] == '副詞' or 
            wclass[0] == '名詞'
            #wclass[0] == '連体詞'
        ):
            if wclass[6] != '*':
                wakati += wclass[6] + ' '
        node = node.next
    return wakati

#コーパスを分かち書きにする関数
def make_data(pd):
    training = []
    for index in pd.index:
        sent = analyze(pd.at[index,'review'])
        training.append(sent)
    
    return training