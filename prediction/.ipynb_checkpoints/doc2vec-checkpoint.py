import pandas as pd
import numpy as np
import sys
#!pip install mecab-python3
#!pip install googletrans
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from utils import preprocess

    
# Doc2Vecのモデルを保存
def save_model(words_list, titles_list, model_path):
    trainings = [TaggedDocument(words = words ,tags = [titles_list[i]]) for i, words in enumerate(words_list)]
    # save model
    model = Doc2Vec(documents=trainings, dm=1, vector_size=200, alpha=0.025, min_count=1)
    model.save(model_path)
    
# Doc2Vecにより類似度のnumpyリストを出力
def predict_movies(input_list, topn=1005, model_path='data/doc2vec1005id.model'):
    model = Doc2Vec.load(model_path)
    vec = model.infer_vector(input_list)
    array = np.array(model.docvecs.most_similar([vec], topn=topn))
    return np.ravel(array[array[:,0].argsort(), :][:,1:])