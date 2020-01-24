import sys
import numpy as np
import pandas as pd
from prediction import lsi
from prediction import doc2vec
from prediction import fasttext
from utils.preprocess import preprocess_TextToList
from prediction import query_expansion


def search_for_movies(query, topn=10, w_review = 1, w_syn = 1, w_r_d=1, w_r_f=1, w_r_l=1, w_r_t=1):
    title_list = pd.read_csv("data/movies.csv")['title'].values.tolist()
    rate_list = pd.read_csv("data/movies.csv")['rate'].values.tolist()
    # query_list = preprocess_TextToList(query, stopwords_path='data/stop_words_review.csv')
    query_list = query_expansion.expansion_magic(query, 8, stopwords_path='data/stop_words_review.csv')
    # lsi
    review_pred = lsi.predict_movies(query_list, corpus_path='data/lsiReviewCorpus.txt',
                       dic_path='data/lsiReviewDic.dict')
   # review_pred = doc2vec.predict_movies(input_list = query_list, topn=1005, model_path='data/doc2vec_reviews.model')
    syn_pred = lsi.predict_movies(query_list, corpus_path='data/lsiSynCorpus.txt', dic_path='data/lsiSynDic.dict')

    # fasttext

    review_pred += fasttext.predict(model_path='data/fasttext_review', word_list=query_list)
    syn_pred    += fasttext.predict(model_path='data/fasttext_synopsis', word_list=query_list)

    pred_list = w_review * review_pred + w_syn * syn_pred

#    review_pred = doc2vec.predict_movies(input_list = query_list, topn=1005, model_path='data/doc2vec_reviews.model')

    sorted_id = np.argsort(pred_list)[::-1]
    return np.array(title_list)[np.array(sorted_id)][:topn], np.array(rate_list)[np.array(sorted_id)][:topn]
