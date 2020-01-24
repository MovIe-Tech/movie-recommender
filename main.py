import sys
import numpy as np
import pandas as pd
from utils.preprocess import preprocess_TextToList
from prediction import lsi
from prediction import doc2vec
from prediction import query_expansion


args = sys.argv
input_text = args[1]
input_list = preprocess_TextToList(input_text)
# input_list = query_expansion.expansion_magic(input_text, 5)  # クエリ拡張機能の実装、引数５は５倍拡張すると言う意味

pred_list = lsi.predict_movies(input_list)
titles_list = pd.read_csv("data/movies.csv")['title'].values.tolist()
sorted_id = np.argsort(pred_list)[::-1]
print(f"{titles_list[sorted_id[0]]} : {pred_list[sorted_id[0]]}")
print(f"{titles_list[sorted_id[1]]} : {pred_list[sorted_id[1]]}")
print(f"{titles_list[sorted_id[2]]} : {pred_list[sorted_id[2]]}")
print(f"{titles_list[sorted_id[3]]} : {pred_list[sorted_id[3]]}")
print(f"{titles_list[sorted_id[4]]} : {pred_list[sorted_id[4]]}")
# print(lsi.predict_movies(input_list))
# print(doc2vec.predict_movies(input_list, topn=2000, model_path='data/doc2vec1005id.model'))