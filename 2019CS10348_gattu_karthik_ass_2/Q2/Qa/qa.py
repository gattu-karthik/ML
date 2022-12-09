#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import sys 
import pandas as pd 
import time 
# import pickle
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


# In[121]:


train_path = str(sys.argv[1])+'/train_data.pickle'
test_path = str(sys.argv[2])+'/test_data.pickle'

# train_path = "../../part2_data/train_data.pickle"
# test_path = "../../part2_data/test_data.pickle"

 
train_data = pd.read_pickle(train_path)
test_data = pd.read_pickle(test_path)


# In[122]:


# d = 8, so take classes 3 & 4.
X_train = []
Y_train = []
X_test = []
Y_test = []
# print(len(test_data['labels']))
# print(len(train_data['labels']))
for i in range(len(train_data['labels'])) :
    if(train_data['labels'][i]==3):
        X_train.append(train_data['data'][i].reshape((3072,)))
        Y_train.append([-1])
    elif(train_data['labels'][i]==4):
        X_train.append(train_data['data'][i].reshape((3072,)))
        Y_train.append([1])


for i in range(len(test_data['labels'])) :
    if(test_data['labels'][i]==3):
        X_test.append(test_data['data'][i].reshape((3072,)))
        Y_test.append([-1])
    elif(test_data['labels'][i]==4):
        X_test.append(test_data['data'][i].reshape((3072,)))
        Y_test.append([1])
        
X_train = np.array(X_train)
X_train = X_train/255
Y_train = np.array(Y_train)

X_test = np.array(X_test)
X_test = X_test/255
Y_test = np.array(Y_test)


# In[123]:


print(X_train.shape)
print(Y_train.shape)


# In[62]:


#-------------------------data-loading. and all shaping  done till here-------------------------#


# In[128]:


def train_linear_svm() :
    start_time = time.time()
    C = 1.0
    X_1 = Y_train*X_train
    H = np.dot(X_1,X_1.T)
    m = X_train.shape[0]

    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-1*np.ones((m,1)))
    G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h = cvxopt_matrix(np.vstack((np.zeros((m,1)), np.ones((m,1))*C)))
    A = cvxopt_matrix(Y_train.reshape(1,-1))*1.0
    b = cvxopt_matrix(np.zeros(1))

    cvxopt_solvers.options['show_progress'] = False
    solution = cvxopt_solvers.qp(P,q,G,h,A,b) 

    alpha = np.array(solution['x'])
    w = np.dot((Y_train*alpha).T, X_train).reshape(-1,1)
    S = (alpha > 1e-4).flatten()

    b = Y_train[S] - np.dot(X_train[S], w)
    b = np.mean(b)

    end_time = time.time()

    return (w,S,b,end_time-start_time)
  
    
  

    
    


# In[129]:


w,S,b,training_time = train_linear_svm()


# In[130]:


print(training_time)


# In[137]:


def get_images():
    images = X_train[S]
    for i in range(5):
        temp = (images[i].reshape((32,32,3)))
        plt.imshow(temp)
        plt.savefig("Img_"+str(i)+".png")


# In[138]:


get_images()


# In[141]:


def get_W_image() :
    W_image =w.flatten()
    max_w=max(W_image)
    min_w=min(W_image)
    W_image=(W_image-min_w)/(max_w-min_w)
    W_image=W_image.reshape((32,32,3))
    plt.imshow(W_image)
    plt.savefig("w_image.png")


# In[142]:


get_W_image()


# In[161]:


count=0
file = open("sv_in_linear.txt","w")
for i in S:
    if(i):
        file.write("TRUE\n")
        count += 1
    else:
        file.write("FALSE\n")
file.close()
print(count)


# In[147]:


def get_prediction() :
    d = np.dot(X_test,w)+b
    d = np.sign(d)
    return d


# In[153]:


Y_pred = get_prediction()


# In[156]:


def get_accuracy():
    correct_preds=0
    for i in range(len(Y_test)):
        if(Y_test[i]==Y_pred[i]):
            correct_preds += 1
    accuracy=(correct_preds/len(Y_test))*100
    return accuracy


# In[157]:


accuracy  = get_accuracy()


# In[158]:


print("Printing w :")
print(w)
print("Printing b :")
print(b)
print("Time taken for Training =")
print("Test_Accuracy =" + str(accuracy))

