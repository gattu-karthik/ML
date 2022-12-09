#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import numpy as np
import pandas as pd
import math
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import sys



###read data

###apply algoritm

##ploting the outputs


# In[2]:


path_to_train_data = str(sys.argv[1])
# path_to_train_data = "../../part1_data/train/"
train_data_pos = glob.glob(path_to_train_data  + '/pos/*.txt')
train_data_neg = glob.glob(path_to_train_data  + '/neg/*.txt')


# In[3]:


total_words_in_vocabulary=0
total_pos_words=0
total_neg_words=0
dic_of_pos_words=dict()
dic_of_neg_words=dict()


# In[4]:


for p in train_data_pos  :
    f  = open(p,"r")
    review = f.read()
    f.close()
    words_in_review  = review.split()
    total_pos_words = total_pos_words +  len(words_in_review)
    for word in words_in_review  :
        dic_of_pos_words[word] = dic_of_pos_words.get(word,0)+1
        
total_words_in_vocabulary = total_words_in_vocabulary  + len(dic_of_pos_words)

for n in train_data_neg  :
    f  = open(n,"r")
    review = f.read()
    f.close()
    words_in_review  = review.split()
    total_neg_words = total_neg_words +  len(words_in_review)
    for word in words_in_review  :
        if(dic_of_pos_words.get(word,0)==0) :
            total_words_in_vocabulary = total_words_in_vocabulary + 1
        dic_of_neg_words[word] = dic_of_neg_words.get(word,0)+1



log_of_pos = math.log1p(len(train_data_pos))
log_of_neg = math.log1p(len(train_data_neg))


# In[5]:


def get_prediction(text) :
    p_pred=log_of_pos
    n_pred=log_of_neg
    p_total_log = math.log1p(total_words_in_vocabulary+total_pos_words)
    n_total_log = math.log1p(total_words_in_vocabulary+total_neg_words)
    words = text.split() 
    for word in words :
        p_pred += math.log1p(1+dic_of_pos_words.get(word,0))
        p_pred -= p_total_log
        n_pred += math.log1p(1+dic_of_neg_words.get(word,0))
        n_pred -= n_total_log
    
    if p_pred >= n_pred :
        return "pos"
    else :
        return "neg"
            
    
    


# In[6]:


# ---------------------------test accuracy----------------------#
path_to_test_data = str(sys.argv[2])
# path_to_test_data =  "../../part1_data/test/"
test_data_pos = glob.glob(path_to_test_data  + '/pos/*.txt')
test_data_neg = glob.glob(path_to_test_data  + '/neg/*.txt')


# In[7]:


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
print(len(test_data_pos))
print(len(test_data_neg))
print(accuracy)
fn = len(test_data_pos) - tp
fp = len(test_data_neg) - tn
conf_mat = np.array([[tn,fn],[fp,tp]])
print(conf_mat)


# In[8]:


# ---------------------------train accuracy----------------------#
tp = 0
tn = 0
for p in train_data_pos :
    f = open(p,"r")
    review = f.read()
    f.close()
    if (get_prediction(review) == "pos") : 
        tp += 1

for n in train_data_neg :
    f = open(n,"r")
    review = f.read()
    f.close()
    if (get_prediction(review) == "neg") : 
        tn += 1
        
accuracy = (tp + tn)/(len(train_data_pos)+len(train_data_neg))
print(accuracy)


# In[9]:


# -----------------------------------------part_a_ii----------------------------------------#

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





# In[10]:


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


# In[ ]:

def random_prediction():
    tp=0
    tn=0
    for i in range(len(test_data_pos)):
        rand_val = np.random.random()
        if (rand_val >= 0.5): tp += 1
       
    for i in range(len(test_data_neg)) :
        rand_val = np.random.random()
        if (rand_val < 0.5): tn += 1

    return tp,tn


# In[26]:


tp,tn = random_prediction()
fn = len(test_data_pos) - tp
fp = len(test_data_neg) - tn
conf_mat = np.array([[tn,fn],[fp,tp]])
print(tp)
print(tn)
print(conf_mat)
print((tp+tn)/(len(test_data_pos)+len(test_data_neg)))


# In[27]:


majority_pred_accuracy = (max(len(test_data_neg), len(test_data_pos))/(len(test_data_pos)+len(test_data_neg)))*100
print(majority_pred_accuracy)
conf_mat = np.array([[0,0],[5000,10000]])
print(conf_mat)






