import sys
import numpy as np
import pandas as pd
from prediction import lsi
from prediction import doc2vec
from utils.preprocess import preprocess_TextToList


def search_for_movies(query, topn, w_r_d=1, w_r_f=1, w_r_l=1, w_r_t=1):
    query_list = preprocess_TextToList(query)
    review_pred = doc2vec.predict_movies(input_list = query_list, topn=1005, model_path='data/doc2vec_review.model')