from natto import MeCab
import load_stopwords as ls
def txt2words(txt) -> list:
    posid = [36,37,38,40,41,42,43,44,45,46,47,50,51,52,66,67,2,31,36,10,34]
    words = []
    parser = MeCab()
    nodes = parser.parse(txt,as_nodes=True)
    for node in nodes:
        if not node.is_eos():
            feature = node.feature.split(',')
            if node.posid in posid and feature[6] != "*":
                words.append(feature[6]) 
    return words

def clean(word_list) -> list:
    stws = ls.load_stopwords()
    return [word for word in word_list if word not in stws]