'''

NAIVE-BAYES CLASSIFIER FOR DOCUMENT CLASSIFIER

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

'''
  
# %% LOAD LIBRARIES

import os
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm

from matplotlib import style
from matplotlib import rcParams
from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as esw

import bayes_tools

# %% PLOT SETTINGS

plt.style.use(['science','ieee'])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cm"],
    "mathtext.fontset": "cm",
    "font.size": 24})

# %% FUNCTION DEFINITIONS

def fit_NBclassifier(trainset, trainlabel):
    nbclassifier = MultinomialNB()
    nbclassifier.fit(trainset, trainlabel)
    
    return nbclassifier

# %% IMPORT DATA

data = pd.read_csv(r'data/sentiment_analysis.csv')

test_train_ratio = 0.7
corpus_train, corpus_test, label_train, label_test = train_test_split( \
    data['text'].to_list(), data['class'].to_list(), test_size=test_train_ratio)

method = 'tfidf'
# method = 'bog'

if method == 'tfidf':
    vectorizer = TfidfVectorizer(tokenizer=str.split, stop_words=esw)

elif method == 'bog':
    vectorizer = CountVectorizer(tokenizer=str.split, stop_words=esw)

corpus_train_mat = vectorizer.fit_transform(corpus_train)
corpus_train_mat = corpus_train_mat.toarray()

corpus_test_mat = vectorizer.transform(corpus_test)
corpus_test_mat = corpus_test_mat.toarray()


# %% TRAIN

NB_clf = fit_NBclassifier(corpus_train_mat, label_train)
label_predicted = NB_clf.predict(corpus_test_mat)
accuracy = accuracy_score(label_test, label_predicted)

# %% TEST

conf_mat = confusion_matrix(label_test, label_predicted)*2/corpus_test_mat.shape[0]
labels = sorted(set(label_predicted))

# %% PLOT

os.makedirs('./results/ex4', exist_ok=True)
path = './results/ex4/'

save_res = path + 'conf_mtx_' + method + '_train_test_split_' + str(test_train_ratio)

bayes_tools.plot_confusion_matrix(conf_mat, save=save_res)

# %%
