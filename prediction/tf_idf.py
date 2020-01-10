import pandas as pd
import os
import sys
import MeCab

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# データフレームを表示するときカラムを省略しない
pd.set_option('display.max_columns', None)
# 浮動小数点を表示するときは小数点以下 2 桁で揃える
pd.options.display.float_format = '{:0.2f}'.format

#コーパスの全単語のtfidfのデータフレームを返す
def tf_idf(corpus):
    # scikit-learn の TF-IDF 実装
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(corpus)

    # TF-IDF を表示する
    df = pd.DataFrame(data=X_tfidf.toarray(),
                      columns=tfidf_vectorizer.get_feature_names())
    return df

#その単語のtfidtを上から出力する関数    
def word_tf_idf(review_df,tfidf_df,word):
    sorted=tfidf_df.loc[:,[word]].sort_values(word,ascending=False)
    for i in list(sorted.index):
        if tfidf_df.at[i,word] == 0:
            break
        print(str(review_df.at[i,'title'])+' : '+str(tfidf_df.at[i,word]))
        
#文書を分かち書きにする関数
def analyze(text):
    tagger = MeCab.Tagger("-Ochasen")
    tagger.parse('')
    node = tagger.parseToNode(text)
    wakati = ''
    while node:
        word = node.surface
        wclass = node.feature.split(',')
        #mecabは以下の順でnode.featureに情報を入れる　[0]品詞,[1]品詞細分類1,[2]品詞細分類2,[3]品詞細分類3,[4]活用形,[5]活用型,[6]原形,[7]読み,[8]発音
        if (
            #wclass[0] == '記号' or
            wclass[0] == '形容詞' or
            #wclass[0] == '助詞' or
            #wclass[0] == '助動詞' or
            #wclass[0] == '接続詞' or
            #wclass[0] == '接頭詞' or
            wclass[0] == '動詞' or
            #wclass[0] == '副詞' or 
            wclass[0] == '名詞'
            #wclass[0] == '連体詞'
        ):
            if wclass[6] != '*':
                wakati += wclass[6] + ' '
        node = node.next
    return wakati

#コーパスを分かち書きにする関数
def make_data(pd):
    training = []
    for index in pd.index:
        sent = analyze(pd.at[index,'review'])
        training.append(sent)
    
    return training
    
review_df = pd.read_csv("1005data.csv")
review_df = review_df.rename(columns={'Unnamed: 0':'index', 'Title':'title', 'Rating Score':'score', 'Reviews':'review'}) 

tfidf_df = tf_idf(make_data(review_df))
word_tf_idf(review_df,tfidf_df,"冒険")
