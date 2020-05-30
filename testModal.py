from __future__ import print_function, division
import pickle
from future.utils import iteritems
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import nltk
import numpy as np
from sklearn.utils import shuffle

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

wordnet_lemmatizer = WordNetLemmatizer()

# from http://www.lextek.com/manuals/onix/stopwords1.html
stopwords = set(w.rstrip() for w in open('stopwords.txt'))


def my_tokenizer(s):
    s = s.lower()  # downcase
    tokens = nltk.tokenize.word_tokenize(s)  # split string into words (tokens)
    # remove short words, they're probably not useful
    tokens = [t for t in tokens if len(t) > 2]
    # put words into base form
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]  # remove stopwords
    return tokens


positive_reviews = ["Rajesh is a good boy"]
# create a word-to-index map so that we can create our word-frequency vectors later
# let's also save the tokenized versions so we don't have to tokenize again later
word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []
orig_reviews = []

for review in positive_reviews:
    orig_reviews.append(review)
    tokens = my_tokenizer(review)
    print("Tokens:: ", tokens)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1
print(word_index_map)

# # now let's create our input matrices


def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1)  # last element is for the label
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum()  # normalize it before setting label
    x[-1] = label
    # print("xxxxx:: ", x)
    return x


N = len(positive_tokenized)
# (N x D+1 matrix - keeping them together for now so we can shuffle more easily later
data = np.zeros((N, len(word_index_map) + 1))
i = 0
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    print("XY::: ", xy)
    data[i, :] = xy
    i += 1

orig_reviews, data = shuffle(orig_reviews, data)

X = data[:, :-1]
print("#########", X)

filename = 'finalized_model.sav'

loaded_model = pickle.load(open(filename, 'rb'))
preds = loaded_model.predict(X)
# # preds1 = model.predict(['Good', 'boy'])
print('!@#!#@R@#%$#', preds)
# P = model.predict_proba(X)[:, 1]  # p(y = 1 | x)
