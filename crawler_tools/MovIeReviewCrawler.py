import requests
import csv
from bs4 import BeautifulSoup
import bs4
import pandas as pd
import re
import time
import pickle
import numpy as np

# 大域変数　途中でもデータが取れるため
reviews_df = pd.DataFrame([], columns=["title", "rate", "reviews"])

def get_page_urls(initial_page, pages):
    # 映画のURLリストを取る関数
    # 引数：　開始ページ、総ページ数
    #　返り値：　映画URLのリスト
    urls = []
    for i in range(initial_page, initial_page + pages):
        r = requests.get("https://movies.yahoo.co.jp/movie/?page="+str(i))
        soup = BeautifulSoup(r.content, "html.parser")
        tags = soup.find_all("li",{"class":"col"})
        for tag in tags:
            url="https://movies.yahoo.co.jp/movie/"+tag.get("data-cinema-id")+"/"
            urls.append(url)
    return urls

def get_review_urls(movie_url, reviews_num):
    print("in get_review_urls")
    #　映画ごとのレビューのURLリストを取る関数
    # 引数：　映画のURL, 取りたい映画ごとのレビューの数(10の倍数で良い)
    #　返り値：　レビューURLのリスト
    review_urls = []
    for p in range(1, int(1 + reviews_num/10)):
        url  =  movie_url + f"review/?sort=mrf&page={p}"
        try:
            r = requests.get(url)
            soup = BeautifulSoup(r.content, "html.parser")
            review_soup = soup.find_all("a", class_="listview__element--right-icon")  

            for link in review_soup:
                review_urls.append("https://movies.yahoo.co.jp" + link.get("href"))
        except:
            print("biggest page: ", p-1)
            
    print("out get_review_urls")
    return review_urls

def get_reviews(movie_url, reviews_num):
    print("in get_reviews")
    #　映画ごとのレビューリストを取る関数
    # 引数：　映画のURL, 取りたい映画ごとのレビューの数(10の倍数で良い)
    #　返り値：　レビューのリスト
    review_urls = get_review_urls(movie_url, reviews_num)
    reviews = []
    for url in review_urls:
        r = requests.get(url)
        time.sleep(0.1)
        soup = BeautifulSoup(r.content, "html.parser")
        try: 
            contents = soup.find("p", class_="text-small text-break text-readable p1em").contents
            review = ""
            for c in contents:
                if type(c) == bs4.element.NavigableString:
                    review = review + c
            reviews.append(review)
        except:
            print("error in get_reviews")
    print("out get_reviews")
    return reviews

def get_movies_reviews(urls_list, reviews_per_movie):
    print("in get_movies_reviews")
    # 複数の映画のレビューを取って、dataframeを返す関数。
    # 引数：　映画URLのリス、映画ごとの取りたいレビューの数。
    # 返り値：　映画レビューdataframe
    global reviews_df
    
    num = reviews_per_movie
    for url in urls_list:
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        title = re.sub(r" -(.)*","",soup.title.contents[0])
        
        try:
            rate = soup.find("span", class_="rating-score").find("span").contents[0]
        except AttributeError:
            rate = ""
            print("non-scr:", url)
            
        reviews = get_reviews(url, num)
        
        addRow = pd.DataFrame([title, rate, reviews], index=reviews_df.columns).T
        reviews_df = reviews_df.append(addRow)
    print("out get_movies_reviews")
    return reviews_df

def crawl_movies_reviews(initial_page, pages, reviews_per_movie):
    # 映画のレビューを取って、最終のdataframeを返す関数。
    # 引数：　映画の開始ページ、総ページ数。
    # 返り値：　映画レビューdataframe
    global reviews_df
    movie_pages = get_page_urls(initial_page, pages)
    print(len(movie_pages)," movie page urls:")
    for page in movie_pages:
        print(page)
    get_movies_reviews(movie_pages, reviews_per_movie)
    reviews_df = reviews_df.reset_index(drop=True)
    return reviews_df
