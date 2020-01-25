import sys
import numpy as np
import pandas as pd
from prediction import lsi
from prediction import doc2vec
from prediction import fasttext
from utils.preprocess import preprocess_TextToList
from prediction import query_expansion
from prediction import TFIDF


def search_for_movies(query, topn=10, w_review=1, w_syn=1, w_rl=1, w_sl=1, w_rd=1, w_sd=1, w_rf=1, w_sf=1, w_rt=1, w_st=1):
    title_list = pd.read_csv("data/movies.csv")['title'].values.tolist()
    rate_list = pd.read_csv("data/movies.csv")['rate'].values.tolist()
    
    # remove here after inplement query expansion
    query_list_review = preprocess_TextToList(query, stopwords_path='data/stop_words_review.csv')
    query_list_syn = preprocess_TextToList(query, stopwords_path='data/stop_words_synopsis.csv')
    
    # query expansion
    #query_list_review = query_expansion.expansion_magic(query, 8, stopwords_path='data/stop_words_review.csv')
    #query_list_syn = query_expansion.expansion_magic(query, 8, stopwords_path='data/stop_words_synopsis.csv')
    
    # lsi
    review_pred = w_rl * lsi.predict_movies(query_list_review, corpus_path='data/lsiReviewCorpus.txt', dic_path='data/lsiReviewDic.dict')
    syn_pred = w_sl * lsi.predict_movies(query_list_syn, corpus_path='data/lsiSynCorpus.txt', dic_path='data/lsiSynDic.dict')
    
    #doc2vec
    review_pred += w_rd * doc2vec.predict_movies(input_list = query_list_review, model_path='data/doc2vecReview.model')
    syn_pred += w_sd * doc2vec.predict_movies(input_list = query_list_syn, model_path='data/doc2vecSynopsis.model')

    # fasttext
    review_pred += w_rf * fasttext.predict(model_path='data/fasttext_review', word_list=query_list_review)
    syn_pred    += w_sf * fasttext.predict(model_path='data/fasttext_synopsis', word_list=query_list_syn)
    
    # TFIDF
    review_pred += w_rt * TFIDF.TFIDF_pred_review(query = query_list_review, csv_path = 'data/917datafin.csv')
    syn_pred    += w_st * TFIDF.TFIDF_pred_synopsis(query = query_list_syn, csv_path = 'data/917datafin.csv')

    pred_list = w_review * review_pred + w_syn * syn_pred

    sorted_id = np.argsort(pred_list)[::-1]
    return np.array(title_list)[np.array(sorted_id)][:topn], np.array(rate_list)[np.array(sorted_id)][:topn]
