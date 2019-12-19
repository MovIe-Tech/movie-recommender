import pandas as pd
import numpy as np
import sys
#!pip install mecab-python3
#!pip install googletrans
from googletrans import Translator
import MeCab
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

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
    
def save_model(data_path, stopwords_path, model_path):
    ## read csv
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
    without_stopwords = [word for word in splitted_reviews if word not in stopwords_list]
    without_stopwords = []
    for i in range(len(splitted_reviews)):
        without_stopwords.append([word for word in splitted_reviews[i] if word not in stopwords_list])
    trainings = [TaggedDocument(words = words ,tags = [titles_list[i]]) for i, words in enumerate(without_stopwords)]
    # save model
    model = Doc2Vec(documents=trainings, dm=1, vector_size=200, alpha=0.025, min_count=1)
    model.save(model_path)

def preprocess(text, stopwords_path='data/stopwords.csv'):
    splitted_reviews = analysis(text).split(' ')
    stopwords = pd.read_csv(stopwords_path, encoding='utf-8').T.values.tolist()[0]
    return [word for word in splitted_reviews if word not in stopwords]

def predict_movies(text, model_path='doc2vec_neologd.model'):
    model = Doc2Vec.load(model_path)
    vec = model.infer_vector(preprocess(text))
    return model.docvecs.most_similar([vec])
    
def preprocess_translation(japanese_list):
    translator = Translator()
    predict_movies("心温まる")
    translations = [word.text for word in translator.translate(japanese_list)]
    return [word.text for word in translator.translate(translations, dest='ja')] 