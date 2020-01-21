import sys
import numpy as np
import pandas as pd
from prediction import lsi
from prediction import doc2vec
from utils.preprocess import preprocess_TextToList


def search_for_movies(query):
    query_list = preprocess_TextToList(query)