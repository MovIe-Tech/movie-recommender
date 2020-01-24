import sys
import numpy as np
import pandas as pd
from prediction import lsi
from prediction import doc2vec
from utils.preprocess import preprocess_TextToList


def search_for_movies(query, topn=10, w_r_d=1, w_r_f=1, w_r_l=1, w_r_t=1):
    title_list = pd.read_csv("data/movies_data.csv")['title'].values.tolist()
    rate_list = pd.read_csv("data/movies_data.csv")['rate'].values.tolist()
    query_list = preprocess_TextToList(query)
    review_pred = predict_movies(query_list, corpus_path='/data/lsiReviewCorpus.txt',
                       dic_path='/data/lsiReviewDic.dict', stopwords_path='/data/stop_words_review.csv')
   # review_pred = doc2vec.predict_movies(input_list = query_list, topn=1005, model_path='data/doc2vec_reviews.model')
    syn_pred = predict_movies(query_list, corpus_path='/dada/lsiSynCorpus.txt', dic_path='/data/lsiSynDic.dict', 
                                 stopwords_path='/data/stop_words_synopsis.csv')
    
    pred_list = review_pred + syn_pred
    sorted_id = np.argsort(pred_list)[::-1]
    return np.array(title_list)[np.array(sorted_id)][:topn], np.array(rate_list)[np.array(sorted_id)][:topn]