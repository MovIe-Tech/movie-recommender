import requests
from bs4 import BeautifulSoup
import time
import re
import csv
import pandas as pd

def get_urls(start=1, end=1):
    """
    get movie urls from the start page to end page
    and return the urls list
    """
    urls = []
    yahoo_url = 'https://movies.yahoo.co.jp/movie/'
    
    for page in range(start, end+1):
        temp_url = yahoo_url + '?page=' + (str)(page)
        source = requests.get(temp_url).text
        soup = BeautifulSoup(source, features='lxml')
        time.sleep(1)
        try:
            matches = soup.find('div', id='lst', class_='tracked_mods')
            matches = matches.find_all('li', class_='col')
        except Exception:
            matches = None
        
        for match in matches:
            movie_code = match.a['href']
            urls.append('https://movies.yahoo.co.jp'+movie_code)
            
    return urls

def get_revw_urls(url, revwNum=10):
    """
    get all review urls upon to revwNum, and return the urls list
    """
    revw_urls = []
    page = 1
    
    while revwNum > 0:
        page_url = url + 'review/?sort=mrf&page' + (str)(page)
        source = requests.get(page_url).text
        soup = BeautifulSoup(source, features='lxml')
        time.sleep(1)
        try:
            revw_code = soup.find('div', id='revwlst')
            revw_code = revw_code.find_all('li', class_=None)
            
            # append every review url to revw_urls
            for i in range(min(10, revwNum)):
                try:
                    code = revw_code[i].a['href']
                    temp_url = 'https://movies.yahoo.co.jp' + code
                except Exception:
                    temp_url = None
                revw_urls.append(temp_url)
                
        except Exception:
            pass
            
        page = page + 1   # junp to next page
        revwNum = revwNum - 10
        
    return revw_urls

def get_reviews(urls, synopsis=False, revwNum=10):
    """
    get detailed information from all urls
    and save them to pandas.DataFrame and return it
    """
    # set necessary field names
    if synopsis == False:
        columns = ['Title', 'Rating Score', 'Reviews']
    else: 
        columns = ['Title', 'Rating Score', 'Synopsis', 'Reviews']
    
    # database
    count = 0
    movies = pd.DataFrame(columns=columns)
    
    for url in urls:
        # get movie title
        source = requests.get(url).text
        soup = BeautifulSoup(source, features='lxml')
        time.sleep(1)
        title = soup.h1.span.text
        
        # get movie rating-score
        try:
            score = soup.find('span', class_='rating-score').span.text
        except Exception:
            score = None
        
        # get synopsis if necessary
        if synopsis:
            try:
                synopsis_url = url + 'story/'
                source = requests.get(synopsis_url).text
                soup = BeautifulSoup(source, features='lxml')
                synopsis_text = soup.find_all('section', class_='section')[2].p.text
            except Exception:
                synopsis_text = None
        
        # get movie reviews
        reviews = []
        revw_urls = get_revw_urls(url, revwNum=revwNum)
        
        for url in revw_urls:
            try:
                source = requests.get(url).text
                soup = BeautifulSoup(source, features='lxml')
                time.sleep(1)
                review = soup.article.find('div', id='revwdtl').p.text

                # delete unnecessary labels
                review = re.sub('\\n', '', review) 
                review = re.sub('<br/>', '', review)
            except Exception:
                review = None
            reviews.append(review)
        
        # append all the detailed information to DataFrame
        if synopsis == True:
            movies.loc[count] = [title] + [score] + [synopsis_text] + [reviews]
        else: 
            movies.loc[count] = [title] + [score] + [reviews]
        count = count + 1
    
    return movies

urls = get_urls(81, 83)
data = get_reviews(urls, synopsis=True, revwNum=17)
data.to_csv('data_2.csv')
