import requests
from bs4 import BeautifulSoup

def load_stopwords() -> list:
    slothlib_path = "http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt"
    r = requests.get(slothlib_path)
    raw = str(BeautifulSoup(r.content, "html.parser"))
    stopwords = raw.split()
    return stopwords

