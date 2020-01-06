import pandas as pd
import numpy as np
import sys
#!pip install mecab-python3
#!pip install googletrans
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from utils.preprocess import preprocess_TextToList

    
# Doc2Vecのモデルを保存
def save_model(words_list, titles_list, model_path):
    trainings = [TaggedDocument(words = words ,tags = [titles_list[i]]) for i, words in enumerate(words_list)]
    # save model
    model = Doc2Vec(documents=trainings, dm=1, vector_size=200, alpha=0.025, min_count=1)
    model.save(model_path)
    
# Doc2Vecにより
def predict_movies(text, model_path='doc2vec523_neologd.model', stopwords_path='../data/stopwords.csv'):
    model = Doc2Vec.load(model_path)
    vec = model.infer_vector(preprocess_TextToList(text, stopwords_path))
    return model.docvecs.most_similar([vec])