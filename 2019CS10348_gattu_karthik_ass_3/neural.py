#!/usr/bin/env python
# coding: utf-8

# In[671]:


import  pandas as pd
import numpy as np
import time
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


# In[672]:


# train_data_path =
# test_data_path =
# output_folder_path =

train_data_path = "COL774_fmnist"
test_data_path = "COL774_fmnist"
# output_folder_path = 


# In[673]:


#-----------------------Reading. the data----------------------
train_data = pd.read_csv(train_data_path+"/fmnist_train.csv",header=None)
test_data = pd.read_csv(test_data_path+"/fmnist_test.csv",header=None)
# print(len(train_data))
# print(len(test_data))


# In[674]:


# print(train_data)
train_data = train_data.sample(frac=1)
# print(train_data.shape)


# In[675]:


Y_train = train_data.iloc[:,-1]
Y_test = test_data.iloc[:,-1]
# print(Y_train.shape)
# print(Y_test.shape)


# In[676]:


X_train = train_data.drop(train_data.columns[-1],axis=1)
X_test = test_data.drop(test_data.columns[-1],axis=1)
# print(X_train.shape)
# print(X_test.shape)


# In[677]:


X_validation = X_train.iloc[0:int(0.15*len(X_train))]
Y_validation  = Y_train.iloc[0:int(0.15*len(Y_train))] 

X_train  = X_train.iloc[int(0.15*len(X_train)) :]
Y_train = Y_train.iloc[int(0.15*len(Y_train)) :]

X_validation = X_validation/255
X_train = X_train/255
X_test = X_test/255

# print(len(X_validation))
# print(len(Y_validation))
# print(len(X_train))
# print(len(Y_train))


# In[678]:


# ------------------------one-hot encoding--------------------------
Y_train = pd.get_dummies(Y_train).to_numpy()
Y_validation = pd.get_dummies(Y_validation).to_numpy()
Y_test = pd.get_dummies(Y_test).to_numpy()
# print(Y_train.shape)
# print(Y_validation.shape)
# print(Y_test.shape)


# In[679]:


# ------------------------adding  bias term--------------------------
X_train = np.concatenate((X_train.to_numpy(),np.ones((len(X_train),1))),axis=1)
X_test = np.concatenate((X_test.to_numpy(),np.ones((len(X_test),1))),axis=1)
X_validation = np.concatenate((X_validation.to_numpy(),np.ones((len(X_validation),1))),axis=1)


# In[680]:


# print(X_train.shape)
# print(X_test.shape)
# print(X_validation.shape)


# In[681]:


# ---------------------------creating mini-batches--------------------
mini_batch_size = 100
lr = 0.1
mini_batches=[]
for i in range(0,len(X_train),mini_batch_size):
    mini_batches.append(np.hstack((X_train[i:i+mini_batch_size,:],Y_train[i:i+mini_batch_size])))


# In[682]:


# print(len(mini_batches))


# In[683]:


# print(len(mini_batches))
# print(mini_batches[0].shape)
# print(mini_batches[0][1].shape)


# In[684]:


def activation(act_function,x):
    if (act_function=="Sigmoid"):
        return 1/(1+np.exp(-x))
    elif(act_function == "ReLU") :
        return np.maximum(0.0, x)
    


# In[685]:


def  derivative_of_activation(act_function,x):
    if (act_function=="Sigmoid"):
        return x*(1-x)
    elif(act_function == "ReLU") :
        x[x<=0] = 0
        x[x>0] = 1
        return x
    


# In[686]:


# -------------Initialization of theta---------------
# M-batch size
# n-no:of featrues
# hid_lyr_arch = [,,,,]
# r-no:of targetclasses
def normal_initialize_theta(n,hid_lyr_arch,r):
    theta=[]
    hid_lyr_arch.append(r)
    arch = [n-1]+hid_lyr_arch
#     print("architechure")
#     print(arch)
#     print("architechure len = "+str(len(arch)))
    for i in range(len(arch)-1):
        theta.append(np.random.normal(0,0.05,(arch[i]+1,arch[i+1])))
    return theta
        
        
    


# In[689]:


# ------------------Forward Propagation------------------
def forward_propagation(theta,X,act_function):
    fd_prop = []
    fd_prop.append(X)
#     print(X.shape)
# #     print(theta)
# #     print(fd_prop.shape)
#     print(fd_prop)
    
    for i in range(len(theta)):
#         print("fd_prob")
#         print(fd_prop[i])
        if (i == len(theta)-1):
            fd_prop.append(activation("Sigmoid",np.dot(fd_prop[i],theta[i])))
        else :
            fd_prop.append(activation(act_function,np.dot(fd_prop[i],theta[i])))
            fd_prop[i+1]=np.hstack((np.ones((fd_prop[i+1].shape[0],1)),fd_prop[i+1]))
    return fd_prop
            


# In[690]:


# ------------------Backward Propagation------------------
def backward_propagation(theta,Y,act_function,cost_function,fd_prop):
    bkw_prop = [None for k in range(len(fd_prop))]
    if (cost_function == "MSE"):
        for i in range(len(fd_prop)-1,0,-1):
            if(i==len(fd_prop)-1):
                bkw_prop[i] = ((1/mini_batch_size)*(Y - fd_prop[i])*fd_prop[i]*(1-fd_prop[i]))
            elif(i+1 != len(fd_prop)-1):
                bkw_prop[i] = np.dot(bkw_prop[i+1][:,1:],theta[i].T)*derivative_of_activation(act_function,fd_prop[i])
            else:
                bkw_prop[i] = np.dot(bkw_prop[i+1],theta[i].T)*derivative_of_activation(act_function,fd_prop[i])
                
    else :
        for i in range(len(fd_prop)-1,0,-1):
            if(i==len(fd_prop)-1):
                bkw_prop[i] = ((1/mini_batch_size)*((Y/fd_prop[i])-((1-Y)/(1-fd_prop[i])))*fd_prop[i]*(1-fd_prop[i]))
            elif(i+1 != len(fd_prop)-1):
                bkw_prop[i] = np.dot(bkw_prop[i+1][:,1:],theta[i].T)*derivative_of_activation(act_function,fd_prop[i])
            else:
                bkw_prop[i] = np.dot(bkw_prop[i+1],theta[i].T)*derivative_of_activation(act_function,fd_prop[i])
                
        
    return bkw_prop
        


# In[691]:


def get_cost(X,Y,theta,act_function,cost_function):
    fd_prop = forward_propagation(theta,X,act_function)
    m = X.shape[0]
    if(cost_function == "Cross_Entropy"):
        a = (Y*np.log(fd_prop[-1]))
        b = ((1-Y)*(np.log(1-fd_prop[-1])))
        return -(1/m)*(np.sum((a+b)))
    elif (cost_function == "MSE"):
        a = np.sum((Y-fd_prop[-1])**2)
        return (1/(2*m))*a


# In[692]:


def get_accuracy(X,Y,theta,act_function,need_mat):
    temp_preds = forward_propagation(theta,X,act_function)[-1]
    preds = np.argmax(temp_preds,axis=1)
    actual_preds = np.argmax(Y,axis=1)
#     print(preds.shape)
#     print(actual_preds.shape)
    count = 0
    for i in range(len(Y)):
        if (preds[i]==actual_preds[i]) :
            count += 1
    count = count/len(Y)
    count = count*100
    
    if(need_mat):
        cnfs_mat = confusion_matrix(actual_preds,preds)
        return cnfs_mat,count
    else :
        return count
    


# In[707]:


def start_training(theta,act_function,cost_function,constant_lr,X_validation,Y_validation):
    start_time = time.time()
    epoch = 1 
    condition_count = 0
    cost_prev = 0 
    cost_now = 0
    threshold = 1e-5
    lr = 0.1
    if(constant_lr):
        while(1):
#             ---------stopping conditon after each epoch
            if(epoch > 1) :
                if(act_function == "Sigmoid"):
                    if(condition_count == 5):
                        break
                    if(abs(cost_now-cost_prev)<threshold) :
                        condition_count += 1
                elif(act_function == "ReLU"):
                    if(condition_count == 15):
                        break
                    if(abs(cost_now-cost_prev)<threshold) :
                        condition_count += 1
                        
            for batch  in mini_batches :
                X = batch[:,0:len(batch[0])-10]
                Y = batch[:,len(batch[0])-10 : len(batch[0])]
#                 print(X.shape)
#                 print(Y.shape)
        #  -------forward propagation
                fd_prop = forward_propagation(theta , X , act_function)
        #   --------backward propagtion
                bkw_prop = backward_propagation(theta,Y,act_function,cost_function,fd_prop)
        #   -----------theta update
                for i in range(len(theta)):
                    if i != len(theta)-1 :
                        theta[i] += lr*np.dot(fd_prop[i].T, bkw_prop[i+1])[:,1:]  
                    else :
                        theta[i] += lr*np.dot(fd_prop[i].T, bkw_prop[i+1]) 
            
            
            if(epoch==1):
                cost_prev = get_cost(X_validation,Y_validation,theta,act_function,cost_function)
                cost_now = cost_prev
            elif(epoch > 1):
                cost_prev = cost_now
                cost_now = get_cost(X_validation,Y_validation,theta,act_function,cost_function)
            
            epoch += 1
            
   
    
    else :
        lr_0  = lr
        while(1):
        # ---------learning - rate update
            lr = lr_0 / (epoch**0.5)
        #---------stopping conditon after each epoch
            if(epoch > 1) :
                if(act_function == "Sigmoid"):
                    if(condition_count == 5):
                        break
                    if(abs(cost_now-cost_prev)<threshold) :
                        condition_count += 1
                elif(act_function == "ReLU"):
                    if(condition_count == 15):
                        break
                    if(abs(cost_now-cost_prev)<threshold) :
                        condition_count += 1
                
                        
            for batch  in mini_batches :
                X = batch[:,0:len(batch[0])-10]
                Y  = batch[:,len(batch[0])-10 : len(batch[0])]
        #  -------forward propagation
                fd_prop = forward_propagation(theta , X , act_function)
        #   --------backward propagtion
                bkw_prop = backward_propagation(theta,Y,act_function,cost_function,fd_prop)
        #   -----------theta update
                for i in range(len(theta)):
                    if i != len(theta)-1 :
                        theta[i] += lr*np.dot(fd_prop[i].T, bkw_prop[i+1])[:,1:]  
                    else :
                        theta[i] += lr*np.dot(fd_prop[i].T, bkw_prop[i+1]) 
            
            if(epoch==1):
                cost_prev = get_cost(X_validation,Y_validation,theta,act_function,cost_function)
                cost_now = cost_prev
            elif(epoch > 1):
                cost_prev = cost_now
                cost_now = get_cost(X_validation,Y_validation,theta,act_function,cost_function)
            
            epoch += 1
            
            
            
            
    return theta , epoch-1 ,time.time()-start_time
        


# In[696]:


def part_b():
    
    hidden_layer_units = [5, 10, 15, 20, 25]
    accuracy_train = []
    accuracy_test = []
    train_times = []
    
    for i in range(len(hidden_layer_units)):
        theta = normal_initialize_theta(X_train.shape[1],[hidden_layer_units[i]],10)
        theta,epochs,train_time = start_training(theta,"Sigmoid","MSE",True,X_validation,Y_validation)
        
        train_acc = get_accuracy(X_train,Y_train,theta,"Sigmoid",False) 
        c_mat,test_acc = get_accuracy(X_test,Y_test,theta,"Sigmoid",True)
        
        accuracy_train.append(train_acc)
        accuracy_test.append(test_acc)
        train_times.append(train_time)
        print(c_mat)
        
    print(accuracy_train)
    print(accuracy_test)
    print(train_times)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Accuracy (vs) no: of hidden layer units @ 1 hidden_layer")
    ax.set_ylabel("Accuracy_%")
    ax.set_xlabel("no : of hidden units")
    ax.plot(hidden_layer_units, accuracy_test, marker='o', label='Test_Accuracy')
    ax.plot(hidden_layer_units, accuracy_train, marker='o', label='Train_Accuracy')
    plt.legend()
    plt.savefig("2_part_b_acc.png")
    #     plt.show()
    plt.close()


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Runtime (vs) no: of hidden layer units @ 1 hidden_layer")
    ax.set_ylabel("Train_time_(sec)")
    ax.set_xlabel("no : of hidden units")
    ax.plot(hidden_layer_units, train_times, marker='o', label='Train_time')
    plt.legend()
    plt.savefig("2_part_b_rt.png")
    #     plt.show()
    plt.close()
    


# In[697]:


part_b()


# In[698]:


def part_c():
    
    hidden_layer_units = [5, 10, 15, 20, 25]
    accuracy_train = []
    accuracy_test = []
    train_times = []
    
    for i in range(len(hidden_layer_units)):
        theta = normal_initialize_theta(X_train.shape[1],[hidden_layer_units[i]],10)
        theta,epochs,train_time = start_training(theta,"Sigmoid","MSE",False,X_validation,Y_validation)
        
        train_acc = get_accuracy(X_train,Y_train,theta,"Sigmoid",False) 
        c_mat,test_acc = get_accuracy(X_test,Y_test,theta,"Sigmoid",True)
        
        accuracy_train.append(train_acc)
        accuracy_test.append(test_acc)
        train_times.append(train_time)
        print(c_mat)
        
        
    print(accuracy_train)
    print(accuracy_test)
    print(train_times)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Accuracy (vs) no: of hidden layer units @ 1 hidden_layer & adaptive")
    ax.set_ylabel("Accuracy_%")
    ax.set_xlabel("no : of hidden units")
    ax.plot(hidden_layer_units, accuracy_test, marker='o', label='Test_Accuracy')
    ax.plot(hidden_layer_units, accuracy_train, marker='o', label='Train_Accuracy')
    plt.legend()
    plt.savefig("2_part_c_acc.png")
#     plt.show()
    plt.close()
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Runtime (vs) no: of hidden layer units @ 1 hidden_layer & adaptive")
    ax.set_ylabel("Train_time_(sec)")
    ax.set_xlabel("no : of hidden units")
    ax.plot(hidden_layer_units, train_times, marker='o', label='Train_time')
    plt.legend()
    plt.savefig("2_part_c_rt.png")
#     plt.show()
    plt.close()

    


# In[699]:


part_c()


# In[716]:


def part_d():
    print("--------------sigmoid-------------")
    theta = normal_initialize_theta(X_train.shape[1],[100,100],10)
    theta,epochs,train_time = start_training(theta,"Sigmoid","MSE",False,X_validation,Y_validation)     
    train_acc = get_accuracy(X_train,Y_train,theta,"Sigmoid",False) 
    c_mat,test_acc = get_accuracy(X_test,Y_test,theta,"Sigmoid",True)
           
    print(train_acc)
    print(test_acc)
    print(train_time)
    print(c_mat)
    
    print("--------------ReLu-------------")
    theta = normal_initialize_theta(X_train.shape[1],[100,100],10)
    theta,epochs,train_time = start_training(theta,"ReLU","MSE",False,X_validation,Y_validation)     
    train_acc = get_accuracy(X_train,Y_train,theta,"ReLU",False) 
    c_mat,test_acc = get_accuracy(X_test,Y_test,theta,"ReLU",True)
           
    print(train_acc)
    print(test_acc)
    print(train_time)
    print(c_mat)


# In[717]:


part_d()


# In[720]:


def part_e():
    sigmoid_test_accuracies = []
    sigmoid_train_accuracies = []
    sigmoid_train_times = []
    
    relu_test_accuracies = []
    relu_train_accuracies = []
    relu_train_times = []
    
    for i in range(2,6):
        architecture = [50 for j in range (i)]
        
        theta = normal_initialize_theta(X_train.shape[1],architecture,10)
        theta,epochs,train_time = start_training(theta,"Sigmoid","MSE",False,X_validation,Y_validation)     
        train_acc = get_accuracy(X_train,Y_train,theta,"Sigmoid",False) 
        test_acc = get_accuracy(X_test,Y_test,theta,"Sigmoid",False)

        sigmoid_train_accuracies.append(train_acc)
        sigmoid_test_accuracies.append(test_acc)
        sigmoid_train_times.append(train_time)
        
        print("sigmoid")
       
        theta = normal_initialize_theta(X_train.shape[1],architecture,10)
        theta,epochs,train_time = start_training(theta,"ReLU","MSE",False,X_validation,Y_validation)     
        train_acc = get_accuracy(X_train,Y_train,theta,"ReLU",False) 
        test_acc = get_accuracy(X_test,Y_test,theta,"ReLU",False)

        relu_train_accuracies.append(train_acc)
        relu_test_accuracies.append(test_acc)
        relu_train_times.append(train_time)
        
        print("relu")
     
    print("--------------sigmoid-------------")
    print(sigmoid_test_accuracies)
    print(sigmoid_train_accuracies)
    print(sigmoid_train_times)
    
    print("--------------ReLu-------------")
    print(relu_test_accuracies)
    print(relu_train_accuracies)
    print(relu_train_times)
    
    hidden_layers = [2,3,4,5]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Accuracy (vs) no: of hidden layers @  adaptive")
    ax.set_ylabel("Accuracy_%")
    ax.set_xlabel("no : of hidden units")
    ax.plot(hidden_layers, sigmoid_test_accuracies, marker='o', label='Test_Accuracy')
    ax.plot(hidden_layers, sigmoid_train_accuracies, marker='o', label='Train_Accuracy')
    plt.legend()
    plt.savefig("2_part_e_acc_sig.png")
#     plt.show()
    plt.close()
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Accuracy (vs) no: of hidden layers @  adaptive")
    ax.set_ylabel("Accuracy_%")
    ax.set_xlabel("no : of hidden units")
    ax.plot(hidden_layers, relu_test_accuracies, marker='o', label='Test_Accuracy')
    ax.plot(hidden_layers, relu_train_accuracies, marker='o', label='Train_Accuracy')
    plt.legend()
    plt.savefig("2_part_e_acc_relu.png")
#     plt.show()
    plt.close()
    


# In[721]:


part_e()


# In[714]:


def part_f():
#     fill the best architecure obtained in the above part
    architecture = [50,50]
    theta = normal_initialize_theta(X_train.shape[1],architecture,10)
    theta,epochs,train_time = start_training(theta,"ReLU","Cross_Entropy",False,X_validation,Y_validation)

    train_acc = get_accuracy(X_train,Y_train,theta,"ReLU",False) 
    test_acc = get_accuracy(X_test,Y_test,theta,"ReLU",False)

    print(train_acc)
    print(test_acc)
    
    
    
    


# In[715]:


part_f()


# In[710]:


def part_g():
    clf = MLPClassifier(hidden_layer_sizes = (50,50), activation = 'relu', solver = 'sgd', 
                     batch_size = 100, learning_rate_init = 0.1, learning_rate = 'constant', max_iter=400,
                     tol=1e-5, verbose=False)
    st = time.time()
    clf.fit(X_train,Y_train)
    et = time.time()
    
    train_acc = 100*clf.score(X_train,Y_train)
    test_acc = 100*clf.score(X_test,Y_test)
    
    print(train_acc)
    print(test_acc)
    print(et-st)
    
    


# In[711]:


part_g()


# In[ ]:




