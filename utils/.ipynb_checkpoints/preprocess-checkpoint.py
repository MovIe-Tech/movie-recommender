import pandas as pd
import numpy as np
import sys
#!pip install mecab-python3
#!pip install googletrans
from googletrans import Translator
import MeCab


## 基本形で分かち書きをする関数
def analysis(text):
    #mecab = MeCab.Tagger("-Ochasen")
    mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    mecab.parse("")
    mecab.parseToNode("dummy")
    node = mecab.parseToNode(text)
    word = ""
    pre_feature = ""
    while node:
         # 名詞、形容詞、動詞、形容動詞であるかを判定する。
        isUsed = "名詞" in node.feature
        isUsed = "形容詞" in node.feature or isUsed
        isUsed = "動詞" in node.feature or isUsed
        isUsed = "形容動詞" in node.feature or isUsed
         # 以下に該当する場合は除外する。（ストップワード）
        isUsed = (not "代名詞" in node.feature) and isUsed
        isUsed = (not "助動詞" in node.feature) and isUsed
        isUsed = (not "非自立" in node.feature) and isUsed
        isUsed = (not "数" in node.feature) and isUsed
        isUsed = (not "人名" in node.feature) and isUsed
        if isUsed:
            word += " {0}".format(node.feature.split(",")[6])
        '''
        if isUsed:
            if ("名詞接続" in pre_feature and "名詞" in node.feature) or ("接尾" in node.feature):
            word += "{0}".format(node.surface)
        else:
        word += " {0}".format(node.surface)
        #print("{0}{1}".format(node.surface, node.feature))
        '''
        pre_feature = node.feature
        node = node.next
    return word[1:]
    
    
# 作品ごとの単語のリストと作品タイトルのリストを出力
def preprocess_reviews(data_path, stopwords_path):
    ## csvの読み込み
    df = pd.read_csv(data_path, encoding='utf-8', dtype={'Rating Score':'float'})
    df = df.dropna(axis=0, how='all', subset=['reviews'])
    ## lists of reviews and titles
    reviews_list = df['reviews'].T.tolist()
    titles_list = df['title'].T.tolist()
    ## stopword
    stopwords_df = pd.read_csv(stopwords_path, encoding='utf-8')
    stopwords_list = stopwords_df.T.values.tolist()[0]
    ## make training data
    splitted_reviews = [analysis(reviews).split(' ') for i, reviews in enumerate(reviews_list)]
    words_list = []
    for i in range(len(splitted_reviews)):
        words_list.append([word for word in splitted_reviews[i] if word not in stopwords_list])
    return words_list, titles_list


# ストップワードを用い、テキストを単語のリストにする, 入力に利用
def preprocess_TextToList(text, stopwords_path='data/stopwords.csv'):
    splitted_reviews = analysis(text).split(' ')
    stopwords = pd.read_csv(stopwords_path, encoding='utf-8').T.values.tolist()[0]
    return [word for word in splitted_reviews if word not in stopwords]


# 単語のリストを日本語->英語->日本語と翻訳する    
def preprocess_translation(japanese_list):
    translator = Translator()
    predict_movies("心温まる")
    translations = [word.text for word in translator.translate(japanese_list)]
    return [word.text for word in translator.translate(translations, dest='ja')] 