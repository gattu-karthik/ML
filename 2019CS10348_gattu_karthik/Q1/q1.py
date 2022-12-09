#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys

import numpy as np
import matplotlib.pyplot as plt
import time

p1 = sys.argv[1]
p2 = sys.argv[2]

x_path_train = p1+'/X.csv'
y_path_train = p1+'/Y.csv'

x_path_test = p2+'/X.csv' 


def get_gradient(X,Y,theta):
    gradient=np.zeros((len(X[0]),1))
    gradient=X.T.dot(X.dot(theta)-Y)
    gradient/=len(Y)
    return gradient

def get_loss(X,Y,theta):
    J=0
    J=np.sum(np.square(Y-X.dot(theta)))
    J/=2*len(Y)
    return J

def get_cost(X,Y,theta_0,theta_1):
    cost = 0
    for i in range(len(Y)):
        cost += (Y[i]-(theta_0 + theta_1*X[i][1]))**2
    cost /= 2*len(Y)
    return cost

# python is pass by refernce
def plot_b(X_original,X,Y,theta):
    X_vals=X_original[:,1]
    h_theta = X.dot(theta)
    plt.plot(X_vals,h_theta,c = "red")
    plt.scatter(X_vals, Y, c="green")
    plt.title("Hypothesis function")
    plt.xlabel("Acidity of Wine")
    plt.ylabel("Density of Wine")
    plt.savefig('q1_b.png')
    plt.show()

def plot_c1(theta_0_vals,theta_1_vals,J_vals):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    
    theta_0_axis = np.arange(0,1,0.01)
    theta_1_axis = np.arange(0,0.0015,0.000015)
    
    plt.title("J value after each iteration")
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    ax.set_zlabel('J value')
    
    theta_0_mesh, theta_1_mesh = np.meshgrid(theta_0_axis, theta_1_axis)
    ax.plot_surface(theta_0_mesh, theta_1_mesh, get_cost(X, Y, theta_0_mesh, theta_1_mesh) ,color = 'violet', alpha = 0.6)

    i=0
    while(i<len(J_vals)):
        ax.scatter(theta_0_vals[i], theta_1_vals[i], J_vals[i], color = 'red')
#         plt.pause(0.2)
        i += int(np.exp(0.01*i))
    plt.savefig('q1_c1.png')
    plt.show()
    plt.close()

def plot_c2(theta_0_vals,theta_1_vals,J_vals):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    
    theta_0_axis = np.arange(-1,3,0.05)
    theta_1_axis = np.arange(-2,2,0.05)
    
    plt.title("J value after each iteration")
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    ax.set_zlabel('J value')
    
    theta_0_mesh, theta_1_mesh = np.meshgrid(theta_0_axis, theta_1_axis)
    ax.plot_surface(theta_0_mesh, theta_1_mesh, get_cost(X, Y, theta_0_mesh, theta_1_mesh) ,color = 'violet', alpha = 0.6)
    
    i=0
    while(i<len(J_vals)):
        ax.scatter(theta_0_vals[i], theta_1_vals[i], J_vals[i], color = 'red')
    #     plt.pause(0.2)
        i += int(np.exp(0.01*i))
    plt.savefig('q1_c2.png')
    plt.show()
    plt.close()

def plot_d(theta_0_vals,theta_1_vals , J_vals,pause_every):

    fig = plt.figure()
    theta_0_axis = np.arange(-1,3,0.05)
    theta_1_axis = np.arange(-2,2,0.05)
    theta_0_mesh, theta_1_mesh = np.meshgrid(theta_0_axis, theta_1_axis)
    
    plt.title("Contour of error")
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    
    plt.contour(theta_0_mesh, theta_1_mesh, get_cost(X, Y, theta_0_mesh, theta_1_mesh), colors='orange')
    i = 0
    while(i < len(J_vals)):
        plt.scatter(theta_0_vals[i], theta_1_vals[i], J_vals[i], color = 'green')
#         if(i%pause_every == 0):
#              plt.pause(0.2)
        i += 1
    plt.savefig('q1_d.png')
#     plt.close()
    plt.show()
    
    
X=np.loadtxt(x_path_train,dtype=float)
Y=np.loadtxt(y_path_train,dtype=float)

X.shape=(len(X),1)
Y.shape=(len(Y),1)

t = np.ones((len(X),1),dtype=float)
# print(t)
X=np.hstack((t,X))
X_original = np.copy(X)
# print(X)
# print(X.shape)
# print(Y.shape)
X[:, 1] -= np.mean(X[:, 1])
X[:, 1]/=np.std(X[:, 1])
# print(np.std(X[:,1]))
# print(np.mean(X[:,1]))
# print(X)
# print(Y)


J_prev = 0
theta = np.zeros((len(X[0]),1))
J = get_loss(X,Y,theta)
itr = 0
# 
eeta = 0.01
delta = 10**(-14)
# 
J_vals = []
theta_0_vals = []
theta_1_vals = []


st=time.time()
while(abs(J-J_prev)>delta):
    theta_0_vals.append(theta[0][0])
    theta_1_vals.append(theta[1][0])
    J_vals.append(J)
    theta = theta-(eeta)*get_gradient(X,Y,theta)
    J_prev = J
    J = get_loss(X,Y,theta)
    itr+=1
et=time.time()

# print(J_vals)


# writhing out put

X_test = np.loadtxt(x_path_test,dtype=float)
X_test.shape=(len(X_test),1)

t = np.ones((len(X_test),1),dtype=float)
X_test = np.hstack((t,X_test))

X_test[:, 1] -= np.mean(X[:, 1])
X_test[:, 1]/=np.std(X[:, 1])

# Y_output = np.dot(X_test,theta)

file=open('result_1.txt',"w+")
for i in range(len(X_test)):
    file.write(str(X_test[i].dot(theta)[0])+"\n")
file.close()




###################

# print("Learning Rate: " + str(eeta))
# print("Theta :")
# print(theta)
# print("Final loss value: " + str(get_loss(X, Y, theta)))
# print("No of iterations to converge: " + str(itr))
# print("Time taken to  converge: " + str(et-st))

# plot_b(X_original,X,Y,theta)

# plot_c1(theta_0_vals,theta_1_vals,J_vals)

# plot_c2(theta_0_vals,theta_1_vals,J_vals)

# For eeta = 0.001
# skip = 50

# For eeta = 0.025
# skip = 5

# For eeta = 0.1
# skip = 1
# pause_every=5
# plot_d(theta_0_vals, theta_1_vals,J_vals,pause_every)


# In[ ]

####################




