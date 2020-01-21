import numpy as np
import sys
from utils.preprocess import preprocess_TextToList
from prediction import lsi
from prediction import doc2vec
from run import search_for_movies


# args = sys.argv
# input_text = args[1]
# input_list = preprocess_TextToList(input_text)
# sim = lsi.predict_movies(input_list)
# sim_sorted = np.sort(sim)[::-1]
# index_sorted = np.argsort(sim)[::-1]
# print(sim_sorted[:5])
# print(index_sorted[:5])
title_list, rate_list = search_for_movies(query="悲しい気持ち")
print(title_list)
print(rate_list)