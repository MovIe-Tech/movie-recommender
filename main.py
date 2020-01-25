import sys
import numpy as np
import pandas as pd
from utils.preprocess import preprocess_TextToList
from prediction import lsi
from prediction import doc2vec
from prediction import query_expansion
import run

args = sys.argv
input_text = args[1]

titles, rates = run.search_for_movies(query=input_text, topn=10)

print(titles)
print(rates)