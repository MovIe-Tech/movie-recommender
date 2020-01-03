# Doc2Vecのあるディレクトリで以下のようにimportする
import Doc2Vec.doc2vec as Mydoc2vec

# modelを保存する(2分程度かかります)
MyDoc2Vec.save_model('data/523_reviews_df.csv', 'data/stopwords.csv', 'doc2vec523_neologd.model')

# 映画をpredict
MyDoc2Vec.predict_movies('怖い映画が見たい', 'doc2vec523_neologd.model')

## ファイルへのパスが誤っている場合があるので, エラーが出たらパスを確認して下さい.