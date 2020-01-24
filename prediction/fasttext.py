import pandas as pd
import numpy as np
import sys
import csv
!pip install mecab-python3
import MeCab
import fasttext
#!pip install gensim
from gensim.models.wrappers.fasttext import FastText
import time
import datetime

def create_traindata(data_path, stopwords_path, save_path, title_path):
    without_stopwords, titles_list, id_list = preprocess_reviews_length(data_path, stopwords_path)
    trainings = ""

    for id, review in zip(id_list, without_stopwords):
        trainings += " __label__" + str(id) + ", " + ' '.join(review) + "\n"
    with open(save_path, mode='w') as f:
        f.write(trainings)

    titles = pd.Series(titles_list)
    titles.to_csv(title_path, header="title", index=False)


def make_model(trainings_path, myFastText_path):
    model = fasttext.train_supervised(input=trainings_path, epoch=1000, dim=200, loss="hs")
    model.save_model(myFastText_path)


def myFastText(data_path, stopwords_path, myFastText_path, title_path):
    # print(datetime.datetime.now())
    create_traindata(data_path, stopwords_path, 'data/testdata.txt', title_path)
    # print(datetime.datetime.now())
    make_model('data/testdata.txt', myFastText_path)
    # print(datetime.datetime.now())


def predict(model_path, word_list, k=1005):

    titles_list = pd.read_csv(title_path, encoding='utf-8')
    model = fasttext.load_model(model_path)
    result = model.predict(word_list, k)
    ## result : [0:id 1:rate][j:word_list][i:ranking]

    for j in range(len(word_list)):
        for i in range(min([k, len(result[0][j])])):
            id = int(result[0][j][i].replace("__label__" , "").replace("," , ""))
            score[id] += result[1][j][i]

    return score
