#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import sys

p1 = sys.argv[1]
x_path_test = p1+'/X.csv' 

def get_loss(X,Y,theta):
    er=0
    er=np.sum(np.square(Y-X.dot(theta)))
    er/=2*len(X)
    return er


# In[34]:


def get_gradient(X,Y,theta):
    gradient=np.zeros((len(X[0]),1))
    gradient=X.T.dot(X.dot(theta)-Y)
    gradient/=len(X)
    return gradient




# In[52]:


def part_c(theta):
#     here it contains the heder ,used skiproow here
    x = np.loadtxt(open("../data/q2/q2test.csv", "rb"),dtype=float, delimiter=",", skiprows=1)
    X = x[:,:2]
    t = np.ones((len(x),1))
    X = np.hstack((t,X))
    Y = x[:,2:]

    theta_original = [[3],[1],[2]]

    original_hypothesis_loss = get_loss(X, Y , theta_original)
    learned_hypothesis_loss = get_loss(X,Y,theta)

    print("Error with original hypothesis : " + str(original_hypothesis_loss))
    print("Error with learned hypothesis : " + str(learned_hypothesis_loss))

    
    


# In[53]:


def part_d(theta_0_vals,theta_1_vals,theta_2_vals) :
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d' )
    
    plt.title("Movment of Theta")
    plt.xlabel('theta 0')
    plt.ylabel('theta 1')
    ax.set_zlabel('theta 2')
    
    ax.set_xlim(0,3)
    ax.set_ylim(0,1.5)
    ax.set_zlim(0,2)
    
    i=1
    while ( i < len(theta_0_vals)) :
        ax.plot([theta_0_vals[i-1],theta_0_vals[i]],[theta_1_vals[i-1],theta_1_vals[i]],[theta_2_vals[i-1],theta_2_vals[i]])
#         if(i%100==0):
#             plt.pause(0.2)
        i+=1
    
#     plt.pause(1)
    plt.savefig('Theta Movement.png')
    plt.show()
    plt.close()
    
    
# generating the datapoints for training
theta = np.array([3,1,2])
x_1 = np.random.normal(3,2,1000000)
x_2 = np.random.normal(-1,2,1000000)
epsilon = np.random.normal(0,2**(0.5),1000000)



y = np.full((1000000,),0)
for i in range (1000000):
    y[i] = theta[0] + theta[1]*x_1[i] + theta[2]*x_2[i]  + epsilon[i]

x_1 = np.reshape(x_1,(1000000,1))
x_2 = np.reshape(x_2,(1000000,1))
Y = np.reshape(y,(1000000,1))

# print(x_1.shape)
# print(x_2.shape)
# print(y.shape)
# print(np.mean(y))
# print(np.mean(x_1))
# print(np.mean(x_2))

# print(np.std(x_1))
# print(np.std(x_2))


X = np.concatenate((x_1,x_2),axis = 1)
t = np.ones((len(X),1))
X = np.hstack((t,X))



############
# df = pd.readcsv("data/q2/q2test.csv")
# x = np.loadtxt(open("sampled_data.csv", "rb"),dtype=float, delimiter=",", skiprows=1)
# X = x[:,:2]
# t = np.ones((len(x),1))
# X = np.hstack((t,X))

# Y = x[:,2:]

#########

# In[32]:


# print(len(Y))
# print(Y)
# print(X.shape)
# print(Y.shape)
# print(X)


# In[47]:


m = len(X)
J_prev = -1
theta = np.zeros((len(X[0]),1))
J = 0 
eeta = 0.001
itr = 0
converged = False

# delta = 10**(-5)
# r = 1
# b = 10000

delta = 10**(-9)
r = 100
b = 10000

# delta = 10**(-7)
# r = 1000000
# b = 1


# delta = 10**(-9)
# r = 10000
# b = 1000

cnt = 0
theta_0_vals = []
theta_1_vals = []
theta_2_vals = []


# In[48]:


while(converged == False):
    for start in range (0, m , r) :
        if(cnt==b):
            J = J/b
            if(J_prev != -1 and abs(J-J_prev) < delta) :
                converged = True
                break
            J_prev = J
            J = 0
            cnt = 0
              
        end = start + r
        theta_0_vals.append(theta[0][0])
        theta_1_vals.append(theta[1][0])
        theta_2_vals.append(theta[2][0])
        X_b = X[start : end]
        Y_b = Y[start : end]
        theta = theta - (eeta)*get_gradient(X_b , Y_b , theta)
        J += get_loss(X_b,Y_b,theta)
        cnt += 1
    itr += 1
        




# writhing out put

# X_test = np.loadtxt(x_path_test,dtype=float)
# X_test.shape=(len(X_test),2)


# t = np.ones((len(X_test),1),dtype=float)
# X_test = np.hstack((t,X_test))

X_test = pd.read_csv(x_path_test,header=None)
X_test = X_test.to_numpy()
t = np.ones((len(X_test),1),dtype=float)
X_test = np.hstack((t,X_test))


# Y_output = np.dot(X_test,theta)

file=open('result_2.txt',"w+")
for i in range(len(X_test)):
    file.write(str(X_test[i].dot(theta)[0])+"\n")
file.close()


# In[49]:


# print("Final converged theta :")
# print(theta)
# print("\nNo of Epochs for convergence: " + str(itr))


# In[50]:


# part_c(theta)


# In[51]:

# part_d(theta_0_vals,theta_1_vals,theta_2_vals)





