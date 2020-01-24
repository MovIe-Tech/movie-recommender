import pandas as pd
import numpy as np
import sys
import csv
#!pip install mecab-python3
import MeCab
import fasttext
#!pip install gensim
from gensim.models.wrappers.fasttext import FastText
import time
import datetime

def create_traindata(data_path, review_path, synopsis_path):
    #words_list, titles_list, id_list = preprocess_reviews_length(data_path, stopwords_path)
    df = pd.read_csv(data_path)
    
    id_list = df['id'].T.tolist()
    reviews_list = []
    synopsis_list = []
    
    for review, synopsis in zip(df['reviews'], df['synopsis']):
        reviews_list.append(review.split(' '))
        synopsis_list.append(synopsis.split(' '))
    
    review = ""
    synopsis = ""
    
    for id, r, s in zip(id_list, reviews_list, synopsis_list):
        review   += " __label__" + str(id) + ", " + ' '.join(r) + "\n"
        synopsis += " __label__" + str(id) + ", " + ' '.join(s) + "\n"
    with open(review_path, mode='w') as f:
        f.write(review)
    with open(synopsis_path, mode='w') as f:
        f.write(synopsis)


def make_model(trainings_path, myFastText_path, e=1000, d=200):
    model = fasttext.train_supervised(input=trainings_path, epoch=e, dim=d, loss="hs")
    model.save_model(myFastText_path)


def myFastText(data_path, FastText_review_path, FastText_synopsis_path):
    # print(datetime.datetime.now())
    create_traindata(data_path, 'data/review.txt', 'data/synopsis.txt')
    # print(datetime.datetime.now())
    make_model('data/review.txt', FastText_review_path)
    # print(datetime.datetime.now())
    make_model('data/synopsis.txt', FastText_synopsis_path, d=100)
    # print(datetime.datetime.now())


def predict(model_path, word_list, k=917):
    
    # titles_list = pd.read_csv(title_path, encoding='utf-8')
    model = fasttext.load_model(model_path)
    result = model.predict(word_list, k)
    ## result : [0:id 1:rate][j:word_list][i:ranking]
    
    score = np.zeros(917)
    
    for j in range(len(word_list)):
        for i in range(min([k, len(result[0][j])])):
            id = int(result[0][j][i].replace("__label__" , "").replace("," , ""))
            score[id] += result[1][j][i]
    
    return score
