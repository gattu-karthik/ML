#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import sys 
import pandas as pd 
import time 
# import pickle
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
    


# In[51]:



train_path = str(sys.argv[1])+'/train_data.pickle'
test_path = str(sys.argv[2])+'/test_data.pickle'

# train_path = "../../part2_data/train_data.pickle"
# test_path = "../../part2_data/test_data.pickle"

train_data = pd.read_pickle(train_path)
test_data = pd.read_pickle(test_path)


# In[52]:


X_train = []
Y_train = []
X_test = []
Y_test = []
# print(len(test_data['labels']))
# print(len(train_data['labels']))
for i in range(len(train_data['labels'])) :
    if(train_data['labels'][i]==3):
        X_train.append(train_data['data'][i].reshape((3072,)))
        Y_train.append([3])
    elif(train_data['labels'][i]==0):
        X_train.append(train_data['data'][i].reshape((3072,)))
        Y_train.append([0])
    elif(train_data['labels'][i]==1):
        X_train.append(train_data['data'][i].reshape((3072,)))
        Y_train.append([1])
    elif(train_data['labels'][i]==2):
        X_train.append(train_data['data'][i].reshape((3072,)))
        Y_train.append([2])
    elif(train_data['labels'][i]==4):
        X_train.append(train_data['data'][i].reshape((3072,)))
        Y_train.append([4])
                       

for i in range(len(test_data['labels'])) :
    if(test_data['labels'][i]==3):
        X_test.append(test_data['data'][i].reshape((3072,)))
        Y_test.append([3])
    elif(test_data['labels'][i]==0):
        X_test.append(test_data['data'][i].reshape((3072,)))
        Y_test.append([0])
    elif(test_data['labels'][i]==1):
        X_test.append(test_data['data'][i].reshape((3072,)))
        Y_test.append([1])
    elif(test_data['labels'][i]==2):
        X_test.append(test_data['data'][i].reshape((3072,)))
        Y_test.append([2])
    elif(test_data['labels'][i]==4):
        X_test.append(test_data['data'][i].reshape((3072,)))
        Y_test.append([4])
        
X_train = np.array(X_train)
X_train = X_train/255
Y_train = np.array(Y_train)

X_test = np.array(X_test)
X_test = X_test/255
Y_test = np.array(Y_test)


# In[32]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[53]:


tr_data = np.hstack((X_train,Y_train))
# print(tr_data.shape)


# In[37]:


# print(tr_data.shape)
# temp_data = tr_data[np.ix_((tr_data[:, 3072] == 2) | (tr_data[:, 3072] == 1))]
# temp_X_train = np.array(temp_data[:,0:3072])
# temp_Y_train = np.array(temp_data[:,3072:3073])


# In[38]:


# print(temp_data.shape)
# print(temp_X_train.shape)
# print(temp_Y_train.shape)


# In[36]:


print(tr_data.shape)


# In[54]:


Y_preds=[[0 for i in range(5)] for j in range(len(Y_test))]
final_test_preds =  [0 for i in range(len(Y_test))]


# In[55]:


def get_gaussian_kernel(X, Y):
    gamma = 0.001
    m = Y.shape[0]
    n = X.shape[0]
    xx = np.dot(np.sum(np.power(X, 2), 1).reshape(n, 1), np.ones((1, m)))
    yy = np.dot(np.sum(np.power(Y, 2), 1).reshape(m, 1), np.ones((1, n)))     
    return np.exp(-gamma*(xx + yy.T - 2*np.dot(X, Y.T)))


# In[63]:


def gaussian_multiclass():
    total_training_time=0
    for i in range(5):
        for j in range(i):
            st = time.time()
            temp_data = tr_data[np.ix_((tr_data[:, 3072] == i) | (tr_data[:, 3072] == j))]
            temp_X_train = np.array(temp_data[:,0:3072])
            temp_Y_train = np.array(temp_data[:,3072:3073])
            
            for l in range(len(temp_Y_train)):
                if (temp_Y_train[l][0] == i):
                    temp_Y_train[l][0] = 1
                else:
                    temp_Y_train[l][0] = -1
            
            C = 1.0
            m = temp_X_train.shape[0]

            K = get_gaussian_kernel(temp_X_train, temp_X_train)
            P = cvxopt_matrix(np.multiply(K,np.dot(temp_Y_train,temp_Y_train.T)))
            q = cvxopt_matrix(-np.ones((m, 1)))
            G = cvxopt_matrix(np.vstack((-np.eye((m)), np.eye(m))))
            h = cvxopt_matrix(np.vstack((np.zeros((m,1)), np.ones((m,1))*C)))
            A = cvxopt_matrix(temp_Y_train.reshape((1, -1)))
            A = cvxopt_matrix(A, (1, m), 'd')
            # A = A.astype('float')
            b = cvxopt_matrix(0.0)


            cvxopt_solvers.options['show_progress'] = False
            solution = cvxopt_solvers.qp(P, q, G, h, A, b)
            
            alpha = np.array(solution['x'])
            S = (alpha > 1e-4).flatten()
            alpha = alpha[S]
            sv_X = temp_X_train[S]
            sv_Y = temp_Y_train[S]
            b = np.mean(sv_Y - np.sum(get_gaussian_kernel(sv_X, sv_X) * alpha * sv_Y, axis=0))
            
            total_training_time = total_training_time + time.time() - st
            
            temp_pred = np.sum(get_gaussian_kernel(sv_X, X_test) * alpha * sv_Y, axis=0) + b
            temp_pred = np.sign(temp_pred)
            
            for l in range(len(temp_pred)):
                if (temp_pred[l] == 1):
                    Y_preds[l][i] += 1
                else: 
                    Y_preds[l][j] += 1
    
    for i in  range(len(Y_preds)):
        final_test_preds[i] = np.argmax(Y_preds[i])
        
    return total_training_time
    
    


# In[64]:


t_time = gaussian_multiclass()
print("Total time taken for training = "+str(t_time))


# In[65]:


count=0
for i in range(len(Y_test)):
    if(Y_test[i]==final_test_preds[i]):
        count += 1
print("Accuracy on  Test_data  = "+str(count/len(Y_test)))


# In[66]:


labels = np.unique(Y_test)
confusion_matrix = np.zeros((len(labels), len(labels)))

final_test_preds = np.array(final_test_preds)
final_test_preds = final_test_preds.reshape(Y_test.shape)
# print(predicted_rating.shape)
for i in range(len(labels)):
    for j in range(len(labels)):
        confusion_matrix[i, j] = np.sum((Y_test == labels[i]) & (final_test_preds == labels[j]))
print(confusion_matrix)

