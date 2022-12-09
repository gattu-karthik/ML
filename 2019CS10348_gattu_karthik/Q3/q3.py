#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys

p1 = sys.argv[1]
p2 = sys.argv[2]

x_path_train = p1+'/X.csv'
y_path_train = p1+'/Y.csv'

x_path_test = p2+'/X.csv' 


# In[39]:


def get_hypothesis(X, theta):
    return 1/(1 + np.exp(-X.dot(theta)))


# In[40]:


def get_likelihood(X, Y, theta) :
    a_1 = np.log(get_hypothesis(X,theta))
    a_0 = np.log(1-get_hypothesis(X,theta))
    return np.sum(np.multiply(Y,a_1)+np.multiply(1-Y,a_0))
    


# In[41]:


def get_gradient(X,Y,theta) :
    gradient = np.zeros((len(X[0]),1))
    gradient = X.T.dot(Y - get_hypothesis(X, theta))
    return gradient


# In[42]:


def get_hessian(X,theta) :
    h = get_hypothesis(X, theta)
    t = np.diagflat(h*(1-h))
    return -X.T.dot(t).dot(X)
    


# In[58]:


def get_prediction(x0,x1_flatten,x2_flatten,theta) :
#     print(x0.shape)
#     print(x1_flatten.shape)
#     print(x2_flatten.shape)
    t = np.hstack((x0,x1_flatten,x2_flatten))
    y = get_hypothesis(t,theta)
    for i in range(len(y)):
        if(y[i] > 0.5):
            y[i] = 1
        else:
            y[i] = 0
    return y
    


# In[59]:


def part_b(X_original) :
    min_x1  = X_original[:,1].min() - 1 
    min_x2  = X_original[:,2].min() - 1 
    max_x1  = X_original[:,1].max() + 1 
    max_x2  = X_original[:,2].max() + 1
    
    x1_grid = np.linspace(min_x1, max_x1, 1000)
    x2_grid = np.linspace(min_x2, max_x2, 1000)
    
    x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
    
    x1_flatten = x1_mesh.flatten()
    x2_flatten = x2_mesh.flatten()
    
    x0 = np.ones((len(x1_flatten),1))
    x1_flatten = np.reshape(x1_flatten, (len(x1_flatten), 1))
    x2_flatten = np.reshape(x2_flatten, (len(x2_flatten), 1))

    x1_flatten = (x1_flatten - np.mean(x1_flatten)) / np.std(x1_flatten)
    x2_flatten = (x2_flatten - np.mean(x2_flatten)) / np.std(x2_flatten)
    
#     print(x0.shape)
#     print(x1_flatten.shape)
#     print(x2_flatten.shape)

    y_predicted = get_prediction(x0,x1_flatten,x2_flatten,theta)
#     print(y_predicted)  

    y_mesh = y_predicted.reshape(x1_mesh.shape)
#     print(x1_mesh.shape)
#     print(x2_mesh.shape)
#     print(y_mesh)
    plt.contour(x1_mesh, x2_mesh, y_mesh, colors = 'black') 
    x_pos_vals = np.array([X_original[i] for i in range(len(X)) if Y[i] == 1])
    x_neg_vals = np.array([X_original[i] for i in range(len(X)) if Y[i] == 0])
    
    plt.title('Decision boundary fit by logistic regression')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    
                  
    plt.scatter(x_pos_vals[:, 1], x_pos_vals[:, 2], color = 'orange', label = '1')
    plt.scatter(x_neg_vals[:, 1], x_neg_vals[:, 2], color = 'blue', label = '0')
    
    plt.legend()
                  
    plt.savefig('decision boundary.png')
    plt.show()
     


# In[ ]:


x = np.loadtxt(open(x_path_train, "rb"),dtype=float, delimiter=",")
t = np.ones((len(x),1))
X = np.hstack((t,x))
X_original = np.copy(X)

X[:,1] = (X[:,1] - np.mean(X[:,1])) / np.std(X[:,1])
X[:,2] = (X[:,2] - np.mean(X[:,2])) / np.std(X[:,2])


Y = np.loadtxt(open(y_path_train, "rb"),dtype=float, delimiter=",")
Y = Y.reshape((len(Y),1))

# print(X.shape)
# print(Y.shape)

# print(X)
# print(Y)



# In[ ]:


itr = 0 
theta = np.zeros((len(X[0]),1))
J_prev = 0
J = get_likelihood(X, Y, theta)
delta = 10**(-12)

while(abs(J - J_prev) > delta) :
    J_prev = J
    H_inverse = np.linalg.inv(get_hessian(X,theta))
    theta = theta - H_inverse.dot(get_gradient(X,Y,theta))
    J = get_likelihood(X,Y,theta)
    itr+=1
    




# writhing out put

# X_test = np.loadtxt(x_path_test,dtype=float)
# X_test.shape=(len(X_test),2)
X_test = pd.read_csv(x_path_test,header=None)
X_test = X_test.to_numpy()

t = np.ones((len(X_test),1),dtype=float)
X_test = np.hstack((t,X_test))

X_test[:,1] = (X_test[:,1] - np.mean(X_original[:,1])) / np.std(X_original[:,1])
X_test[:,2] = (X_test[:,2] - np.mean(X_original[:,2])) / np.std(X_original[:,2])

Y_output = get_hypothesis(X_test,theta)
for i in range(len(Y_output)):
    if(Y_output[i] > 0.5):
        Y_output[i] = 1
    else:
        Y_output[i] = 0


file=open('result_3.txt',"w+")
for i in range(len(Y_output)):
    file.write(str(Y_output[i][0])+"\n")
file.close()

# In[ ]:


# final_loss = get_likelihood(X, Y, theta)
# print("Converged Theta vector:")
# print(theta)
# print("Final loss after convergence: " + str(final_loss))
# print("Total no of iterations to converge: " + str(itr)


# In[60]:


# part_b(X_original)

