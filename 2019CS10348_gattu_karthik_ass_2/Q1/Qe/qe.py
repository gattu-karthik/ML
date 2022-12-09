#!/usr/bin/env python
# coding: utf-8

# In[29]:


# remoove. stop words
# stemming = Merging such variations into a single word
# bigrams. =. instead of using each word as a feature,  treat (two consecutive words) as a feature.
# lemmatization = 
import glob
import numpy as np
import pandas as pd
import math
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import sys


# In[30]:


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# In[31]:


from nltk import bigrams
from nltk import trigrams


# In[32]:


path_to_train_data = str(sys.argv[1])
# path_to_train_data = "../../part1_data/train/"
train_data_pos = glob.glob(path_to_train_data  + '/pos/*.txt')
train_data_neg = glob.glob(path_to_train_data  + '/neg/*.txt')


# In[33]:


total_words_in_vocabulary=0
total_pos_words=0
total_neg_words=0
dic_of_pos_words=dict()
dic_of_neg_words=dict()


# In[34]:


stop_words = stopwords.words('english')


# In[35]:


for p in train_data_pos  :
    f  = open(p,"r")
    review = f.read()
    f.close()
    words_in_review  = review.split()
    stemmer = PorterStemmer()
    
    pos_words = []
#     total_pos_words = total_pos_words +  len(words_in_review)
    for word in words_in_review  :
        word = word.lower()
        if(word not in stop_words):
            word = stemmer.stem(word)
            pos_words.append(word)
            total_pos_words += 1
            dic_of_pos_words[word] = dic_of_pos_words.get(word,0)+1
#      add bigrams. of. pos words
#     pos_words = dic_of_pos_words.keys() 
    pos_word_bigrams = list(bigrams(pos_words))
    
    for word in pos_word_bigrams :
        if(word not in stop_words):
            total_pos_words += 1
            dic_of_pos_words[word] = dic_of_pos_words.get(word,0)+1
        
            
total_words_in_vocabulary = total_words_in_vocabulary  + len(dic_of_pos_words)

for n in train_data_neg  :
    f  = open(n,"r")
    review = f.read()
    f.close()
    words_in_review  = review.split()
    
    neg_words = []
#     total_neg_words = total_neg_words +  len(words_in_review)
    for word in words_in_review  :
        word = word.lower()
        if(word not in stop_words):
            word = stemmer.stem(word)
            neg_words.append(word)
            total_neg_words += 1
            if(dic_of_pos_words.get(word,0)==0) :
                total_words_in_vocabulary = total_words_in_vocabulary + 1
            dic_of_neg_words[word] = dic_of_neg_words.get(word,0)+1

#      add bigrams. of. neg words
#     neg_words = dic_of_neg_words.keys() 
    neg_word_bigrams = list(bigrams(neg_words))
    
    for word in neg_word_bigrams :
        if(word not in stop_words):
            total_neg_words += 1
            if(dic_of_pos_words.get(word,0)==0) :
                total_words_in_vocabulary = total_words_in_vocabulary + 1
            dic_of_neg_words[word] = dic_of_neg_words.get(word,0)+1


# In[36]:


log_of_pos = math.log1p(len(train_data_pos))
log_of_neg = math.log1p(len(train_data_neg))


# In[40]:


def get_prediction(text) :
    p_pred=log_of_pos
    n_pred=log_of_neg
    p_total_log = math.log1p(total_words_in_vocabulary+total_pos_words)
    n_total_log = math.log1p(total_words_in_vocabulary+total_neg_words)
    words = text.split()
    
    cleaned_words = []
    for word in words :
        word = word.lower()
        if(word not in stop_words):
            word = stemmer.stem(word)
            cleaned_words.append(word)
            p_pred += math.log1p(1+dic_of_pos_words.get(word,0))
            p_pred -= p_total_log
            n_pred += math.log1p(1+dic_of_neg_words.get(word,0))
            n_pred -= n_total_log
            
    new_bigrams = list(bigrams(cleaned_words))
    
    for word in new_bigrams :
        if(word not in stop_words):
            p_pred += math.log1p(1+dic_of_pos_words.get(word,0))
            p_pred -= p_total_log
            n_pred += math.log1p(1+dic_of_neg_words.get(word,0))
            n_pred -= n_total_log
            
    
    if p_pred >= n_pred :
        return "pos"
    else :
        return "neg"


# In[41]:


# ---------------------------test accuracy----------------------#
path_to_test_data = str(sys.argv[2])
# path_to_test_data =  "../../part1_data/test/"
test_data_pos = glob.glob(path_to_test_data  + '/pos/*.txt')
test_data_neg = glob.glob(path_to_test_data  + '/neg/*.txt')


# In[49]:


tp = 0
tn = 0
for p in test_data_pos :
    f = open(p,"r")
    review = f.read()
    f.close()
    if (get_prediction(review) == "pos") : 
        tp += 1

for n in test_data_neg :
    f = open(n,"r")
    review = f.read()
    f.close()
    if (get_prediction(review) == "neg") : 
        tn += 1
        
accuracy = (tp + tn)/(len(test_data_pos)+len(test_data_neg))
print(tp)
print(tn)
print(accuracy)
fn = len(test_data_pos) - tp
fp = len(test_data_neg) - tn
conf_mat = np.array([[tn,fn],[fp,tp]])
print(conf_mat)


# In[50]:


precision = tp/(tp + len(test_data_neg) - tn)
recall = tp / (len(test_data_pos))
f1_score = (2*precision*recall)/(precision + recall)

print(precision)
print(recall)
print(f1_score)


# In[51]:


precision = tn/(tn + len(test_data_pos) - tp)
recall = tn / (len(test_data_neg))
f1_score = (2*precision*recall)/(precision + recall)

print(precision)
print(recall)
print(f1_score)


# In[43]:


# -----------------------trigram training------------------------#
for p in train_data_pos  :
    f  = open(p,"r")
    review = f.read()
    f.close()
    words_in_review  = review.split()
    stemmer = PorterStemmer()
    
    pos_words = []
#     total_pos_words = total_pos_words +  len(words_in_review)
    for word in words_in_review  :
        word = word.lower()
        if(word not in stop_words):
            word = stemmer.stem(word)
            pos_words.append(word)
            total_pos_words += 1
            dic_of_pos_words[word] = dic_of_pos_words.get(word,0)+1
#      add trigrams. of. pos words
#     pos_words = dic_of_pos_words.keys() 
    pos_word_trigrams = list(trigrams(pos_words))
    
    for word in pos_word_trigrams :
        if(word not in stop_words):
            total_pos_words += 1
            dic_of_pos_words[word] = dic_of_pos_words.get(word,0)+1
        
            
total_words_in_vocabulary = total_words_in_vocabulary  + len(dic_of_pos_words)

for n in train_data_neg  :
    f  = open(n,"r")
    review = f.read()
    f.close()
    words_in_review  = review.split()
    
    neg_words = []
#     total_neg_words = total_neg_words +  len(words_in_review)
    for word in words_in_review  :
        word = word.lower()
        if(word not in stop_words):
            word = stemmer.stem(word)
            neg_words.append(word)
            total_neg_words += 1
            if(dic_of_pos_words.get(word,0)==0) :
                total_words_in_vocabulary = total_words_in_vocabulary + 1
            dic_of_neg_words[word] = dic_of_neg_words.get(word,0)+1

#      add triigrams. of. neg words
#     neg_words = dic_of_neg_words.keys() 
    neg_word_trigrams = list(trigrams(neg_words))
    
    for word in neg_word_trigrams :
        if(word not in stop_words):
            total_neg_words += 1
            if(dic_of_pos_words.get(word,0)==0) :
                total_words_in_vocabulary = total_words_in_vocabulary + 1
            dic_of_neg_words[word] = dic_of_neg_words.get(word,0)+1


# In[44]:


log_of_pos = math.log1p(len(train_data_pos))
log_of_neg = math.log1p(len(train_data_neg))


# In[45]:


def trigram_prediction(text) :
    p_pred=log_of_pos
    n_pred=log_of_neg
    p_total_log = math.log1p(total_words_in_vocabulary+total_pos_words)
    n_total_log = math.log1p(total_words_in_vocabulary+total_neg_words)
    words = text.split()
    
    cleaned_words = []
    for word in words :
        word = word.lower()
        if(word not in stop_words):
            word = stemmer.stem(word)
            cleaned_words.append(word)
            p_pred += math.log1p(1+dic_of_pos_words.get(word,0))
            p_pred -= p_total_log
            n_pred += math.log1p(1+dic_of_neg_words.get(word,0))
            n_pred -= n_total_log
            
    new_trigrams = list(trigrams(cleaned_words))
    
    for word in new_trigrams :
        if(word not in stop_words):
            p_pred += math.log1p(1+dic_of_pos_words.get(word,0))
            p_pred -= p_total_log
            n_pred += math.log1p(1+dic_of_neg_words.get(word,0))
            n_pred -= n_total_log
            
    
    if p_pred >= n_pred :
        return "pos"
    else :
        return "neg"


# In[46]:


tp = 0
tn = 0
for p in test_data_pos :
    f = open(p,"r")
    review = f.read()
    f.close()
    if (trigram_prediction(review) == "pos") : 
        tp += 1

for n in test_data_neg :
    f = open(n,"r")
    review = f.read()
    f.close()
    if (trigram_prediction(review) == "neg") : 
        tn += 1
        
accuracy = (tp + tn)/(len(test_data_pos)+len(test_data_neg))
print(tp)
print(tn)
print(accuracy)
fn = len(test_data_pos) - tp
fp = len(test_data_neg) - tn
conf_mat = np.array([[tn,fn],[fp,tp]])
print(conf_mat)


# In[47]:


precision = tp/(tp + len(test_data_neg) - tn)
recall = tp / (len(test_data_pos))
f1_score = (2*precision*recall)/(precision + recall)

print(precision)
print(recall)
print(f1_score)


# In[48]:


precision = tn/(tn + len(test_data_pos) - tp)
recall = tn / (len(test_data_neg))
f1_score = (2*precision*recall)/(precision + recall)

print(precision)
print(recall)
print(f1_score)

