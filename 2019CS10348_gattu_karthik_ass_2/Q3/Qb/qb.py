#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import sys 
import pandas as pd 
import time 
# import pickle
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


# In[5]:


from sklearn.svm import SVC


# In[2]:



train_path = str(sys.argv[1])+'/train_data.pickle'
test_path = str(sys.argv[2])+'/test_data.pickle'

# train_path = "../../part2_data/train_data.pickle"
# test_path = "../../part2_data/test_data.pickle"

train_data = pd.read_pickle(train_path)
test_data = pd.read_pickle(test_path)


# In[50]:


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


# In[15]:


tr_data = np.hstack((X_train,Y_train))


# In[16]:


Y_preds=[[0 for i in range(5)] for j in range(len(Y_test))]
final_test_preds =  [0 for i in range(len(Y_test))]


# In[23]:


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
                    
            
            y_train = []
            for l in range(len(temp_Y_train)):
                y_train.append(temp_Y_train[l][0])
            clf=SVC(kernel='rbf',gamma=0.001,C=1)
            clf.fit(temp_X_train,y_train)
            
            total_training_time = total_training_time + time.time() - st
            
            temp_pred = []
            
            for l in range(len(Y_test)):
                t_p = clf.predict([X_test[l]])
                temp_pred.append(t_p)
            
            for l in range(len(temp_pred)):
                if (temp_pred[l] == 1):
                    Y_preds[l][i] += 1
                else: 
                    Y_preds[l][j] += 1
    
    for i in  range(len(Y_preds)):
        final_test_preds[i] = np.argmax(Y_preds[i])
    
    return total_training_time


# In[24]:


t_time = gaussian_multiclass()
print("Total time taken for training = "+str(t_time))


# In[25]:


count=0
for i in range(len(Y_test)):
    if(Y_test[i]==final_test_preds[i]):
        count += 1
print("Accuracy on  Test_data  = "+str(count/len(Y_test)))


# In[26]:


labels = np.unique(Y_test)
confusion_matrix = np.zeros((len(labels), len(labels)))

final_test_preds = np.array(final_test_preds)
final_test_preds = final_test_preds.reshape(Y_test.shape)
# print(predicted_rating.shape)
for i in range(len(labels)):
    for j in range(len(labels)):
        confusion_matrix[i, j] = np.sum((Y_test == labels[i]) & (final_test_preds == labels[j]))
print(confusion_matrix)


# In[71]:


X_test =  []
for i in range(len(test_data['labels'])) :
    if(test_data['labels'][i]==3):
        X_test.append(test_data['data'][i].reshape((3072,)))
       
    elif(test_data['labels'][i]==0):
        X_test.append(test_data['data'][i].reshape((3072,)))
       
    elif(test_data['labels'][i]==1):
        X_test.append(test_data['data'][i].reshape((3072,)))
        
    elif(test_data['labels'][i]==2):
        X_test.append(test_data['data'][i].reshape((3072,)))
        
    elif(test_data['labels'][i]==4):
        X_test.append(test_data['data'][i].reshape((3072,)))
       
        
X_test = np.array(X_test)


# In[72]:


visited  = [-1 for i in range(10)]
for i in range(len(Y_test)):
    predicted = final_test_preds[i]
    actual = Y_test[i]
    
    if(predicted==4 and actual==2  and visited[0] == -1 ):
        visited[0] == 1
        im=(X_test[i].reshape((32,32,3)))
        plt.imshow(im)
        plt.savefig('10.png')
        plt.close()
        
    elif(predicted==2 and actual==4 and visited[1] == -1 ):
        visited[1] == 1
        im=(X_test[i].reshape((32,32,3)))
        plt.imshow(im)
        plt.savefig('1.png')
        plt.close()
    elif(predicted==0 and actual==2 and visited[2] == -1):
        visited[2] == 1
        im=(X_test[i].reshape((32,32,3)))
        plt.imshow(im)
        plt.savefig('2.png')
        plt.close()
    elif(predicted==3 and actual==2 and visited[3] == -1):
        visited[3] == 1
        im=(X_test[i].reshape((32,32,3)))
        plt.imshow(im)
        plt.savefig('3.png')
        plt.close()
    elif(predicted==4 and actual==3 and visited[4] == -1):
        visited[4] == 1
        im=(X_test[i].reshape((32,32,3)))
        plt.imshow(im)
        plt.savefig('4.png')
        plt.close()
    elif(predicted==2 and actual==3 and visited[5] == -1):
        visited[5] == 1
        im=(X_test[i].reshape((32,32,3)))
        plt.imshow(im)
        plt.savefig('5.png')
        plt.close()
    elif(predicted==3 and actual==4 and visited[6] == -1):
        visited[6] == 1
        im=(X_test[i].reshape((32,32,3)))
        plt.imshow(im)
        plt.savefig('6.png')
        plt.close()
    elif(predicted==0 and actual==1 and visited[7] == -1):
        visited[7] == 1
        im=(X_test[i].reshape((32,32,3)))
        plt.imshow(im)
        plt.savefig('7.png')
        plt.close()
    elif(predicted==3 and actual==1 and visited[8] == -1):
        visited[8] == 1
        im=(X_test[i].reshape((32,32,3)))
        plt.imshow(im)
        plt.savefig('8.png')
        plt.close()
    elif(predicted==1 and actual==0 and visited[9] == -1):
        visited[9] == 1
        im=(X_test[i].reshape((32,32,3)))
        plt.imshow(im)
        plt.savefig('9.png')
        plt.close()


# In[51]:


# print(len(final_test_preds))
# print(len(Y_test))
# print(visited)
# print(X_test[1]*225)

