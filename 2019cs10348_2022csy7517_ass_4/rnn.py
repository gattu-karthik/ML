#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# ----------------------------edited------------------------

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# ----------------------------edited------------------------
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
# ----------------------------edited------------------------

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[1]:


import sys
import csv
import pandas as pd
import numpy as np


import matplotlib.image as image
from matplotlib import pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# In[9]:


from PIL import Image

# In[ ]:


root_path  =  sys.argv[1]


# In[5]:


train_x_path = root_path+"/train_x.csv"
test_x_path = root_path+"/non_comp_test_x.csv"
train_y_path = root_path+"/train_y.csv"
# test_y_path = root_path+"/non_comp_test_y.csv"
images_path = root_path+"/images/images/"
# comp_test_path = root_path + "/comp_test_x.csv"


# In[6]:


train_x=pd.read_csv(train_x_path)
test_x=pd.read_csv(test_x_path)
train_y=pd.read_csv(train_y_path)


dataset=[]
dataset_x=train_x['Cover_image_name']
train_title=train_x['Title']
dataset_y=train_y['Genre']


# In[43]:

from torchtext.vocab import GloVe
embedding_glove = GloVe(name='6B', dim=300)

from torchtext.data import get_tokenizer

tokenizer = get_tokenizer("basic_english") 


dataset_test_x=test_x['Title']
#dataset_test=[]
dataset_test_y=test_y['Genre']
dataset_test_emb=[]
for title in dataset_test_x:
    dataset_test_emb.append(embedding_glove.get_vecs_by_tokens(tokenizer(title), lower_case_backup=True))

#for i in range(len(dataset_test_emb)):
    #dataset_test.append((dataset_test_emb[i],dataset_test_y[i]))
    
test_d=[]
for i in range(len(dataset_test_emb)):
    l=len(dataset_test_emb[i])
    test_d.append([l,dataset_test_emb[i], dataset_test_y[i]])
                    
test_d=sorted(test_d, key=lambda data: data[0])   

for i in range(len(dataset_x)):
    dataset.append((dataset_x[i],train_title[i], dataset_y[i]))


# In[44]:


val_dataset=[]
val_dataset_x=test_x['Cover_image_name']
val_title=test_x['Title']
val_dataset_y=test_y['Genre']
for i in range(len(val_dataset_x)):
    val_dataset.append((val_dataset_x[i], val_title[i],val_dataset_y[i]))


# In[45]:


batch_size = 64
val_size = 2000
train_size = len(dataset_x) - val_size 

train_data,val_data = random_split(dataset,[train_size,val_size])


train_d=[]
for i in range(len(train_data)):
    l=len(train_data[i][0])
    train_d.append([l,train_data[i][0], train_data[i][1]])
    #if(l==1):
        #print(train_data[i][0])
    

train_d=[]
for i in range(len(train_data)):
    l=len(train_data[i][0])
    train_d.append([l,train_data[i][0], train_data[i][1]])
    #if(l==1):
        #print(train_data[i][0])


train_d=sorted(train_d, key=lambda data: data[0])

val_d=[]
for i in range(len(val_data)):
    l=len(val_data[i][0])
    val_d.append([l,val_data[i][0], val_data[i][1]])
val_d=sorted(val_d, key=lambda data: data[0])

val_y=[]
for i in range(len(val_d)):
    val_y.append(val_d[i][2])


# In[13]:


batch_size = 16


# In[77]:


train_dl = DataLoader(dataset, batch_size, shuffle = False, num_workers = 2, pin_memory = True)
#train_t_dl=DataLoader(dataset_y, batch_size, shuffle = False, num_workers = 4, pin_memory = True)
val_dl = DataLoader(val_dataset, batch_size,shuffle = False, num_workers = 2, pin_memory = True)


# In[14]:




# In[15]:


transformer= transforms.Compose([
        transforms.Resize(size = (128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# In[89]:


class TitleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TitleRNN, self).__init__()
        #self.emb = nn.Embedding.from_pretrained(glove.vectors)
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc1= nn.Linear(hidden_size*2, 128)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        
        out = self.fc1(out[:, -1, :])
        tl=nn.Tanh()
        out=tl(out)
        out = self.fc2(out)
        out=tl(out)
        return out
    
    def train1(model, device, train_d, epochs):
        learning_r=0.0001

        # Set the model to training mode
        optimizer = optim.Adam(model.parameters(), lr=learning_r)

        # Specify the loss criteria
        loss_criteria = nn.CrossEntropyLoss()

        model.train()
        train_loss = 0
        for epoch in range(epochs):
            if(epoch>20):
                learning_r/=1.5
                optimizer = optim.Adam(model.parameters(), lr=learning_r)
                
            if(epoch>200):
                learning_r/=2
                optimizer = optim.Adam(model.parameters(), lr=learning_r)
                
                
            
            print("Epoch:", epoch)
            i=0
            
            #for l, xb, yb in train_loader:
                #print(xb)
                #print(len(yb))
                #batch_d=[]
                #print(len(xb))
            avg_loss=0
            #print(len(train_d))
            while(i+batch_size < len(train_d)):
                #print(i)
                temp_d=[]
                temp_y=[]
                #if(i>30000):
                    #print(i)
                
                    
                for j in range(batch_size):
                    temp_d.append(train_d[i+j][1])
                    temp_y.append(torch.tensor(train_d[i+j][2]))
                i+=batch_size
                temp_d=pad_sequence(temp_d, batch_first=True)
                #batch_d=torch.stack(temp_d)
                yb=torch.stack(temp_y)
                """
                i=0
                for i in range(len(xb)):
                    #img=image.imread(images_path+xb[i])
                    img=Image.open(images_path+xb[i], 'r')
                    t_img=transformer(img)
                    batch_d.append(t_img)
                
                #print(batch_d)
                batch_d=torch.stack(batch_d)
                #print(batch_d.shape)
                """
                #dataset_x_padded =pad_sequence( xb ,batch_first=True)
                batch_d, yb1 = temp_d.to(device), yb.to(device)
                optimizer.zero_grad()
                #print("Done batching")
                output = model_rnn(batch_d)
                loss = loss_criteria(output, yb1)
                train_loss += loss.item() #
                avg_loss+=train_loss
                loss.backward()
                optimizer.step()
                if(i%10048 == 0):
                    print(i,loss)
                #loss=0
                #print('\tTraining batch Loss: {:.6f}'.format( loss.item()))
                #model.test1(device, val_dl)
            #print("batch")

            avg_loss /=(len(train_d)// batch_size)
            print('Training set: Average loss: {:.6f}'.format(avg_loss))
            model.test1(device, val_d)
            torch.save(model.state_dict(), "a4_model-2-rnn")
        return avg_loss
    
    def test1(model, device,test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        loss_criteria = nn.CrossEntropyLoss()
        with torch.no_grad():
            
            batch_count = 0
            i=0
            while(i+batch_size <= len(test_loader) ):
                temp_d=[]
                temp_y=[]
                #if(i>30000):
                    #print(i)
                for j in range(batch_size):
                    temp_d.append(test_loader[i+j][1])
                    """
                    one_hot_y=[0.0]*30
                    one_hot_y[train_d[i+j][2]-1]=1.0
                    temp_y.append(one_hot_y)
                    """
                    temp_y.append(torch.tensor(test_loader[i+j][2]))
                i+=batch_size
                temp_d=pad_sequence(temp_d, batch_first=True)
                #batch_d=torch.stack(temp_d)
                target=torch.stack(temp_y)
            
                """
                for xb, target in test_loader:
                    batch_count += 1
                    batch_d=[]
                    #print(len(xb))
                    i=0
                    for i in range(len(xb)):
                        #img=image.imread(images_path+xb[i])
                        img=Image.open(images_path+xb[i], 'r')
                        t_img=transformer(img)
                        batch_d.append(t_img)
                    #print(batch_d)
                    batch_d=torch.stack(batch_d)
                    """
                data=temp_d
                #print(data.shape)
                #print(target)
                data, target = data.to(device), target.to(device)

                # Get the predicted classes for this batch
                output = model(data)

                # Calculate the loss for this batch
                test_loss += loss_criteria(output, target).item()
                batch_count+=1

                # Calculate the accuracy for this batch
                _, predicted = torch.max(output.data, 1)
                correct += torch.sum(target==predicted).item()

        # Calculate the average loss and total accuracy for this epoch
        avg_loss = test_loss / batch_count
        print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader),
        100. * correct / len(test_loader)))
    
        # return average loss for the epoch
        return avg_loss
    
    
    def test2(model, device,test_loader):
        model.eval()
        #test_loss = 0
        correct = 0
        #loss_criteria = nn.CrossEntropyLoss()
        with torch.no_grad():
            
            batch_count = 0
            i=0
            while(i+batch_size <= len(test_loader) ):
                temp_d=[]
                temp_y=[]
                #if(i>30000):
                    #print(i)
                for j in range(batch_size):
                    temp_d.append(test_loader[i+j][1])
                    """
                    one_hot_y=[0.0]*30
                    one_hot_y[train_d[i+j][2]-1]=1.0
                    temp_y.append(one_hot_y)
                    """
                    temp_y.append(torch.tensor(test_loader[i+j][2]))
                i+=batch_size
                temp_d=pad_sequence(temp_d, batch_first=True)
                #batch_d=torch.stack(temp_d)
                target=torch.stack(temp_y)
            
                """
                for xb, target in test_loader:
                    batch_count += 1
                    batch_d=[]
                    #print(len(xb))
                    i=0
                    for i in range(len(xb)):
                        #img=image.imread(images_path+xb[i])
                        img=Image.open(images_path+xb[i], 'r')
                        t_img=transformer(img)
                        batch_d.append(t_img)
                    #print(batch_d)
                    batch_d=torch.stack(batch_d)
                    """
                data=temp_d
                #print(data.shape)
                #print(target)
                data, target = data.to(device), target.to(device)

                # Get the predicted classes for this batch
                output = model(data)

                # Calculate the loss for this batch
                test_loss += loss_criteria(output, target).item()
                batch_count+=1

                # Calculate the accuracy for this batch
                _, predicted = torch.max(output.data, 1)
                correct += torch.sum(target==predicted).item()

        # Calculate the average loss and total accuracy for this epoch
        avg_loss = test_loss / batch_count
        print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader),
        100. * correct / len(test_loader)))
    
        # return average loss for the epoch
        return avg_loss
        

# In[90]:


device = "cpu"
if (torch.cuda.is_available()):
    # if GPU available, use cuda (on a cpu, training will take a considerable length of time!)
    device = "cuda"


# In[91]:


model_rnn=TitleRNN(input_size, hidden_size, num_classes)

model_rnn.train1(device=device, train_d=train_d,epochs=500)


y_pred = model_cnn.test1(device=device, test_loader=val_dl)

ans_df=pd.DataFrame(test_x["Id"])
y_comp_pred=[]
for i in range(len(y_pred)):
    y_comp_pred.append(-1)
    
for i in range(len(y_pred)):
    y_comp_pred[i]=int(y_pred[i])
    
ans_df["Genre"]=y_comp_pred
ans_df.to_csv("non_comp_test_pred_y.csv", index=False)







