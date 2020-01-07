import string2words as s2w
import pandas as pd

mdf = pd.read_csv("data/523_reviews_df.csv").drop("Unnamed: 0", axis=1)

sum = 0
for i in range(len(mdf)):
    text = mdf.reviews[i]
    word_list = s2w.clean(s2w.txt2words(text))
    sum += len(word_list)
mean = sum // len(mdf)
print(mean)