#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import sys 
import pandas as pd 
import time 
# import pickle
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg


# In[27]:


from sklearn.svm import SVC


# In[28]:



train_path = str(sys.argv[1])+'/train_data.pickle'
test_path = str(sys.argv[2])+'/test_data.pickle'

# train_path = "../../part2_data/train_data.pickle"
# test_path = "../../part2_data/test_data.pickle"

 
train_data = pd.read_pickle(train_path)
test_data = pd.read_pickle(test_path)


# In[29]:


X_train = []
Y_train = []
X_test = []
Y_test = []
# print(len(test_data['labels']))
# print(len(train_data['labels']))
for i in range(len(train_data['labels'])) :
    if(train_data['labels'][i]==3):
        X_train.append(train_data['data'][i].reshape((3072,)))
        Y_train.append(-1)
    elif(train_data['labels'][i]==4):
        X_train.append(train_data['data'][i].reshape((3072,)))
        Y_train.append(1)


for i in range(len(test_data['labels'])) :
    if(test_data['labels'][i]==3):
        X_test.append(test_data['data'][i].reshape((3072,)))
        Y_test.append(-1)
    elif(test_data['labels'][i]==4):
        X_test.append(test_data['data'][i].reshape((3072,)))
        Y_test.append(1)
        
X_train = np.array(X_train)
X_train = X_train/255
Y_train = np.array(Y_train)

X_test = np.array(X_test)
X_test = X_test/255
Y_test = np.array(Y_test)


# In[40]:


start_time = time.time()
clf=SVC(kernel='linear',C=1)
clf.fit(X_train,Y_train)
end_time = time.time()

print("No of support Vectors for Linear Kernel =" + str(clf.n_support_[0]+clf.n_support_[1]))
print("Training time for Linear Kernel = " + str(end_time-start_time))


# In[42]:


print('w = ',clf.coef_)
print('b = ',clf.intercept_)


# In[32]:


S = clf.support_ 
with open('../Qa/sv_in_linear.txt') as f:
    lines = f.read().splitlines()
matches = 0
for i in range(len(lines)):
    if(lines[i]=="TRUE" and i in S):
        matches += 1
print(matches)


# In[33]:



correct_preds=0
for i in range(len(Y_test)):
    temp = clf.predict([X_test[i]])
    if(Y_test[i]==temp):
        correct_preds += 1
accuracy=(correct_preds/len(Y_test))*100
print(accuracy)


# In[34]:


start_time = time.time()
clf=SVC(kernel='rbf',gamma=0.001,C=1)
clf.fit(X_train,Y_train)
end_time = time.time()

print("No of support Vectors for Linear Kernel =" + str(clf.n_support_[0]+clf.n_support_[1]))
print("Training time for Linear Kernel = " + str(end_time-start_time))


# In[35]:


S = clf.support_ 
with open('../Qb/sv_in_guassian.txt') as f:
    lines = f.read().splitlines()
matches = 0
for i in range(len(lines)):
    if(lines[i]=="TRUE" and i in S):
        matches += 1
print(matches)

