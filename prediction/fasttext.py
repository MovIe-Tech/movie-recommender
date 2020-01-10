import pandas as pd
import numpy as np
import sys
!pip install mecab-python3
import MeCab
import fasttext
#!pip install gensim
from gensim.models.wrappers.fasttext import FastText
import datetime

#fasttext用文書前処理
def create_traindata(data_path, stopwords_path, save_path):
    without_stopwords, titles_list, id_list = preprocess_reviews_length(data_path, stopwords_path)
    trainings = ""

    for id, review in zip(id_list, without_stopwords):
        trainings += " __label__" + str(id) + ", " + ' '.join(review) + "\n"

    with open(save_path, mode='w') as f:
       f.write(trainings)

#モデル生成
def make_model(trainings_path, myFastText_path):
    model = fasttext.train_supervised(input=trainings_path, epoch=2000, dim=300, loss="hs")
    model.save_model(myFastText_path)

def myFastText(data_path, stopwords_path, myFastText_path, titles_path):
    #print(datetime.datetime.now())
    create_traindata(data_path, stopwords_path, 'data/testdata.txt', titles_path)
    make_model('data/testdata.txt', myFastText_path)
    #print(datetime.datetime.now())

#評価
def predict(data_path, model_path, word, k):

    df = pd.read_csv(data_path, encoding='utf-8')
    titles_list = df['title'].T.tolist()

    model = fasttext.load_model(model_path)
    result = model.predict(word, k)
    for i in range(min([k, len(result[0])])):
        print(result[0][i].replace("__label__" , "").replace("," , "") + " : " + str(result[1][i]))
