#!/usr/bin/env python
# coding: utf-8

# In[81]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


import sys

p1 = sys.argv[1]
p2 = sys.argv[2]

x_path_train = p1+'/X.csv'
y_path_train = p1+'/Y.csv'

x_path_test = p2+'/X.csv' 

# In[86]:


def get_P_y(Y,phi):
    return Y*phi + (1-Y)*(1-phi)


# In[87]:


def get_P_xy(x,mu,sigma):
    n = len(x)
    pi = math.pi
    t1 = (x - mu).T.dot(np.linalg.inv(sigma).dot((x - mu)))
    t2 = np.exp(-0.5*t1)
    t3 = (1/(((2*pi)**(n/2))*(np.linalg.det(sigma)**(0.5))))*t2
    return t3[0][0]



# In[88]:


def get_phi(Y) :
    return np.mean(Y)


# In[89]:


def get_mu_0(X, Y) :
    t = X.T.dot(1-Y)
    t=t/np.sum(1-Y)
    t = t.reshape((-1,1))
    return t
    


# In[90]:


def get_mu_1(X, Y) :
    t = X.T.dot(Y)
    t=t/np.sum(Y)
    t = t.reshape((-1,1))
    return t
    


# In[101]:


def get_sigma_0(X, Y, mu_0) :
    n = len(X[0])
    t = np.zeros((n,n))
    for i in range(len(Y)) :
        x = X[i , :]
        x = x.reshape((-1,1))
        if(Y[i]==0):
            t+=(x - mu_0).dot((x-mu_0).T)
    t = t / np.sum(1-Y)
    return t



# In[97]:


def get_sigma_1(X, Y, mu_1) :
    n = len(X[0])
    t = np.zeros((n,n))
    for i in range(len(Y)) :
        x = X[i , :]
        x = x.reshape((-1,1))
        if(Y[i]==1):
            t+=(x - mu_1).dot((x-mu_1).T)
    t = t / np.sum(Y)
    return t
    


# In[100]:


def get_sigma(X, Y, mu_0,mu_1) :
    n = len(X[0])
    t = np.zeros((n,n))
    for i in range(len(Y)) :
        x = X[i , :]
        x = x.reshape((-1,1))
        if(Y[i]==0):
            t+=(x - mu_0).dot((x-mu_0).T)
        else :
            t+=(x - mu_1).dot((x-mu_1).T)
    t = t / len(Y)
    return t


# In[103]:


def get_normalized(x0_flatten,x1_flatten) :
    t = np.hstack((x0_flatten,x1_flatten))
    t[:, 0] = (t[:, 0] - np.mean(t[:, 0]))/np.std(t[:, 0])
    t[:, 1] = (t[:, 1] - np.mean(t[:, 1]))/np.std(t[:, 1])
    return t


# In[104]:


def plot_graph(X_original) :
   

    min_x0  = X_original[:,0].min() - 1 
    min_x1  = X_original[:,1].min() - 1 
    max_x0  = X_original[:,0].max() + 1 
    max_x1  = X_original[:,1].max() + 1

    x0_grid = np.linspace(min_x0, max_x0, 500)
    x1_grid = np.linspace(min_x1, max_x1, 500)

    x0_mesh, x1_mesh = np.meshgrid(x0_grid, x1_grid)

    x1_flatten = x1_mesh.flatten()
    x0_flatten = x0_mesh.flatten()

    x1_flatten = np.reshape(x1_flatten, (len(x1_flatten), 1))
    x0_flatten = np.reshape(x0_flatten, (len(x0_flatten), 1))

    x_nrml = get_normalized(x0_flatten,x1_flatten)

    plt.title('Decision Boundaries of GDA')
    plt.xlabel('Fresh water - x_0')
    plt.ylabel('Marine Water - x_1')

    m = len(x_nrml)
    linear_predictions = np.zeros((m,1))
    quadratic_predictions= np.zeros((m,1))
    
    for i in range (len(x_nrml)) :
        x = x_nrml[i, :]
        x = np.reshape(x, (-1, 1))
        
        lnr_pred_0 = get_P_xy(x, mu_0, sigma)*get_P_y(0, phi)
        lnr_pred_1 = get_P_xy(x, mu_1, sigma)*get_P_y(1, phi)
        
        qd_pred_0 = get_P_xy(x, mu_0, sigma_0)*get_P_y(0, phi)
        qd_pred_1 = get_P_xy(x, mu_1, sigma_1)*get_P_y(1, phi)
        
        if((lnr_pred_0 / (lnr_pred_0 + lnr_pred_1)) < 0.5) :
            linear_predictions[i] = 1
        else :
            linear_predictions[i] = 0
            
        if((qd_pred_0 / (qd_pred_0 + qd_pred_1)) < 0.5) :
            quadratic_predictions[i] = 1
        else :
            quadratic_predictions[i] = 0
        
        
        
        
        
    linear_mesh = linear_predictions.reshape(x1_mesh.shape)
    quadratic_mesh = quadratic_predictions.reshape(x1_mesh.shape)
    
    x_neg = np.array([X_original[i] for i in range(len(X_original)) if Y[i] == 0])
    x_pos = np.array([X_original[i] for i in range(len(X_original)) if Y[i] == 1])
    
    plt.scatter(x_neg[:, 0], x_neg[:, 1], color = 'orange', label = 'Alaska')
    plt.scatter(x_pos[:, 0], x_pos[:, 1], color = 'blue', label = 'Canada')

        
    plt.contour(x0_mesh, x1_mesh, linear_mesh, colors = 'black')
    plt.contour(x0_mesh, x1_mesh, quadratic_mesh, colors = 'brown')

    plt.legend()
    
    plt.savefig('GDA_q4.png')
    plt.show()



# In[82]:


# X = np.loadtxt(open(x_path_train, "rb"))
X = pd.read_csv(x_path_train,header=None)
X=X.to_numpy()
X_original = np.copy(X)


# In[85]:


X[:, 0] = (X[:, 0] - np.mean(X[:, 0])) / np.std(X[:, 0])
X[:, 1] = (X[:, 1] - np.mean(X[:, 1])) / np.std(X[:, 1])


# In[83]:


y = np.loadtxt(open(y_path_train, "rb"),dtype = str)
Y = np.zeros((len(y),1))
for i in range (len(y)) :
    if (y[i]== 'Canada'):
        Y[i] = 1
    else :
        Y[i] = 0 
Y  = Y.reshape((len(Y),1))
# print(Y.shape)
# print(X.shape)


# In[ ]:


phi = get_phi(Y)
mu_0 = get_mu_0(X, Y)
mu_1 = get_mu_1(X, Y)
sigma_0 = get_sigma_0(X, Y, mu_0)
sigma_1 = get_sigma_1(X, Y, mu_1)
sigma = get_sigma(X, Y, mu_0, mu_1)


# writhing out put
X_test = pd.read_csv(x_path_test,header=None)
X_test = X_test.to_numpy()

X_test[:, 0] = (X_test[:, 0] - np.mean(X[:, 0])) / np.std(X[:, 0])
X_test[:, 1] = (X_test[:, 1] - np.mean(X[:, 1])) / np.std(X[:, 1])

Y_output=[]
for i in range (len(X_test)) :
    x = X_test[i]
    x = np.reshape(x, (-1, 1))

    qd_pred_0 = get_P_xy(x, mu_0, sigma_0)*get_P_y(0, phi)
    qd_pred_1 = get_P_xy(x, mu_1, sigma_1)*get_P_y(1, phi)

    if((qd_pred_0 / (qd_pred_0 + qd_pred_1)) < 0.5) :
        Y_output.append('Canada')
    else :
        Y_output.append('Alaska')

file=open('result_4.txt',"w+")
for i in range(len(Y_output)):
    file.write(Y_output[i]+"\n")
file.close()

# In[ ]:


# print("phi: " + str(phi))
# print("\nmu_0:")
# print(mu_0)
# print("\nmu_1:")
# print(mu_1)
# print("\nsigma_0:")
# print(sigma_0)
# print("\nsigma_1:")
# print(sigma_1)
# print("\nsigma:")
# print(sigma)


# In[105]:


# plot_graph(X_original)

    
    

