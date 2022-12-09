#!/usr/bin/env python
# coding: utf-8

# In[41]:


# remoove. stop words
# stemming = Merging such variations into a single word
import glob
import numpy as np
import pandas as pd
import math
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import sys


# In[14]:


# import nltk
# nltk.download('stopwords')


# In[42]:


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# In[43]:


path_to_train_data = str(sys.argv[1])
# path_to_train_data = "../../part1_data/train/"
train_data_pos = glob.glob(path_to_train_data  + '/pos/*.txt')
train_data_neg = glob.glob(path_to_train_data  + '/neg/*.txt')


# In[44]:


total_words_in_vocabulary=0
total_pos_words=0
total_neg_words=0
dic_of_pos_words=dict()
dic_of_neg_words=dict()


# In[45]:


stop_words = stopwords.words('english')


# In[46]:


for p in train_data_pos  :
    f  = open(p,"r")
    review = f.read()
    f.close()
    words_in_review  = review.split()
    stemmer = PorterStemmer()
    
#     total_pos_words = total_pos_words +  len(words_in_review)
    for word in words_in_review  :
        word = word.lower()
        if(word not in stop_words):
            word = stemmer.stem(word)
            total_pos_words += 1
            dic_of_pos_words[word] = dic_of_pos_words.get(word,0)+1
            
total_words_in_vocabulary = total_words_in_vocabulary  + len(dic_of_pos_words)

for n in train_data_neg  :
    f  = open(n,"r")
    review = f.read()
    f.close()
    words_in_review  = review.split()
#     total_neg_words = total_neg_words +  len(words_in_review)
    for word in words_in_review  :
        word = word.lower()
        if(word not in stop_words):
            word = stemmer.stem(word)
            total_neg_words += 1
            if(dic_of_pos_words.get(word,0)==0) :
                total_words_in_vocabulary = total_words_in_vocabulary + 1
            dic_of_neg_words[word] = dic_of_neg_words.get(word,0)+1


# In[47]:


log_of_pos = math.log1p(len(train_data_pos))
log_of_neg = math.log1p(len(train_data_neg))


# In[48]:


def get_prediction(text) :
    p_pred=log_of_pos
    n_pred=log_of_neg
    p_total_log = math.log1p(total_words_in_vocabulary+total_pos_words)
    n_total_log = math.log1p(total_words_in_vocabulary+total_neg_words)
    words = text.split() 
    for word in words :
        word = word.lower()
        if(word not in stop_words):
            word = stemmer.stem(word)
            p_pred += math.log1p(1+dic_of_pos_words.get(word,0))
            p_pred -= p_total_log
            n_pred += math.log1p(1+dic_of_neg_words.get(word,0))
            n_pred -= n_total_log
    
    if p_pred >= n_pred :
        return "pos"
    else :
        return "neg"
            


# In[49]:


# ---------------------------test accuracy----------------------#
path_to_test_data = str(sys.argv[2])
# path_to_test_data =  "../../part1_data/test/"
test_data_pos = glob.glob(path_to_test_data  + '/pos/*.txt')
test_data_neg = glob.glob(path_to_test_data  + '/neg/*.txt')


# In[51]:


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


# In[39]:


# ---------------------------word clouds----------------------#
temp_pos_dic ={}
word_count=0;
for itr in sorted(dic_of_pos_words.items(),key=lambda t: t[1],reverse=True):
    if(word_count>2000):
        break;
    temp_pos_dic[itr[0]]=itr[1]
    word_count += 1

wordcloud = WordCloud(min_word_length=1,background_color='white').generate_from_frequencies(temp_pos_dic)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.savefig('pos_wordcloud.png')


# In[40]:


temp_neg_dic ={}
word_count=0;
for itr in sorted(dic_of_neg_words.items(),key=lambda t: t[1],reverse=True):
    if(word_count>2000):
        break;
    temp_neg_dic[itr[0]]=itr[1]
    word_count += 1

wordcloud = WordCloud(min_word_length=1,background_color='white').generate_from_frequencies(temp_neg_dic)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.savefig('neg_wordcloud.png')

