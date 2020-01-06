import sys
from utils.preprocess import preprocess_TextToList
from prediction import lsi
from prediction import doc2vec


args = sys.argv
input_text = args[1]
input_list = preprocess_TextToList(input_text)
print(lsi.predict_movies(input_list))
# print(doc2vec.predict_movies(input_list, topn=2000, model_path='data/doc2vec1005id.model'))