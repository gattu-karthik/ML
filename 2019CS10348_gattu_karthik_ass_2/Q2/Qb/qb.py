#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys 
import pandas as pd 
import time 
# import pickle
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


# In[2]:


train_path = str(sys.argv[1])+'/train_data.pickle'
test_path = str(sys.argv[2])+'/test_data.pickle'

# train_path = "../../part2_data/train_data.pickle"
# test_path = "../../part2_data/test_data.pickle"

train_data = pd.read_pickle(train_path)
test_data = pd.read_pickle(test_path)


# In[3]:


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


# In[6]:


def get_gaussian_kernel(X, Y):
    gamma = 0.001
    m = Y.shape[0]
    n = X.shape[0]
    xx = np.dot(np.sum(np.power(X, 2), 1).reshape(n, 1), np.ones((1, m)))
    yy = np.dot(np.sum(np.power(Y, 2), 1).reshape(m, 1), np.ones((1, n)))     
    return np.exp(-gamma*(xx + yy.T - 2*np.dot(X, Y.T)))


# In[23]:


def train_gaussian_svm() :
    start_time = time.time()
    C = 1.0
    m = X_train.shape[0]

    K = get_gaussian_kernel(X_train, X_train)
    P = cvxopt_matrix(np.multiply(K,np.dot(Y_train,Y_train.T)))
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(np.vstack((-np.eye((m)), np.eye(m))))
    h = cvxopt_matrix(np.vstack((np.zeros((m,1)), np.ones((m,1))*C)))
    A = cvxopt_matrix(Y_train.reshape((1, -1)))
    A = cvxopt_matrix(A, (1, m), 'd')
    # A = A.astype('float')
    b = cvxopt_matrix(0.0)


    cvxopt_solvers.options['show_progress'] = False
    solution = cvxopt_solvers.qp(P, q, G, h, A, b)

    alpha = np.array(solution['x'])
    S = (alpha > 1e-4).flatten()
    alpha = alpha[S]
    sv_X = X_train[S]
    sv_Y = Y_train[S]

    b = np.mean(sv_Y - np.sum(get_gaussian_kernel(sv_X, sv_X) * alpha * sv_Y, axis=0))
    end_time = time.time()
    #     b = np.mean(b)

    return (S,alpha,b,sv_X,sv_Y,end_time-start_time)

    


# In[24]:


S,alpha,b,sv_X,sv_Y,training_time = train_gaussian_svm()


# In[35]:


print(training_time)


# In[46]:


count=0
file = open("sv_in_guassian.txt","w")
for i in S:
    if(i):
        file.write("TRUE\n")
        count += 1
    else:
        file.write("FALSE\n")
file.close()
print(count)



# In[47]:


with open('../Qa/sv_in_linear.txt') as f:
    lines1 = f.read().splitlines()
with open("sv_in_guassian.txt")  as f:
    lines2 = f.read().splitlines()
matches=0
print(len(lines1))
print(len(lines2))
for i in range(len(lines1)-1):
    if(lines1[i] == "TRUE" and lines2[i] == "TRUE"):
        matches += 1
print(matches)


# In[27]:


def get_gaussian_prediction() :
    d = np.sum(get_gaussian_kernel(sv_X, X_test) * alpha * sv_Y, axis=0) + b
    d = np.sign(d)
    return d


# In[28]:


Y_pred = get_gaussian_prediction()


# In[29]:


def get_accuracy():
    correct_preds=0
    for i in range(len(Y_test)):
        if(Y_test[i]==Y_pred[i]):
            correct_preds += 1
    accuracy=(correct_preds/len(Y_test))*100
    return accuracy


# In[30]:


accuracy  = get_accuracy()


# In[32]:


print("Accuracy on Test set = "+str(accuracy))


# In[25]:


def get_images():
    images = X_train[S]
    for i in range(5):
        temp = (images[i].reshape((32,32,3)))
        plt.imshow(temp)
        plt.savefig("Img_"+str(i)+".png")
    


# In[26]:


get_images()

