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


class IMG_CNN(nn.Module):
  
    def __init__(self, ):
        super(IMG_CNN, self).__init__()
        self.network = nn.Sequential(
                  
                  nn.Conv2d(in_channels=3, out_channels=32, kernel_size = 5), #stride=1, padding =0
                  nn.ReLU(),

                  nn.MaxPool2d(kernel_size =2),

                  nn.Conv2d(in_channels=32,out_channels=64, kernel_size = 5),
                  nn.ReLU(),

                  nn.MaxPool2d(kernel_size =2),
              
                  nn.Conv2d(in_channels=64, out_channels=128, kernel_size = 5),
                  nn.ReLU(),

                  nn.MaxPool2d(kernel_size =2),
                  
                  nn.Flatten(),
                  #nn.Linear((128*(24)*(24)),128), #why not 209?
                  nn.Linear(in_features=18432,out_features=128),
                  nn.ReLU(),
                  nn.Linear(in_features=128, out_features=30)
                  
              )
        
    def forward(self, xb):
        return self.network(xb)
    
    def test1(model, device,test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        loss_criteria = nn.CrossEntropyLoss()
        with torch.no_grad():
            batch_count = 0
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
                data=batch_d
                #print(data.shape)
                #print(target)
                data, target = data.to(device), target.to(device)

                # Get the predicted classes for this batch
                output = model(data)

                # Calculate the loss for this batch
                test_loss += loss_criteria(output, target).item()

                # Calculate the accuracy for this batch
                _, predicted = torch.max(output.data, 1)
                correct += torch.sum(target==predicted).item()

        # Calculate the average loss and total accuracy for this epoch
        avg_loss = test_loss / batch_count
        print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
        # return average loss for the epoch
        return avg_loss

    def train1(model, device, train_loader, epochs):

        # Set the model to training mode
        optimizer = optim.Adam(model.parameters(), lr=0.00001)

        # Specify the loss criteria
        loss_criteria = nn.CrossEntropyLoss()

        model.train()
        train_loss = 0
        for epoch in range(epochs):
            print("Epoch:", epoch)
            i=0
            
            for xb, yb in train_loader:
                #print(xb)
                #print(len(yb))
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
                #print(batch_d.shape)


                batch_d, yb1 = batch_d.to(device), yb.to(device)
                optimizer.zero_grad()
                output = model(batch_d)
                loss = loss_criteria(output, yb1)
                train_loss += loss.item() #
                loss.backward()
                optimizer.step()
                #loss=0
                #print('\tTraining batch Loss: {:.6f}'.format( loss.item()))
                #model.test1(device, val_dl)
            #print("batch")

            avg_loss = train_loss / (i+1)
            print('Training set: Average loss: {:.6f}'.format(avg_loss))
            model.test1(device, val_dl)
            torch.save(model.state_dict(), "a4_model-2")
        return avg_loss
    
    


# In[90]:


device = "cpu"
if (torch.cuda.is_available()):
    # if GPU available, use cuda (on a cpu, training will take a considerable length of time!)
    device = "cuda"


# In[91]:


model_cnn=IMG_CNN().to(device)


y_pred = model_cnn.test1(device=device, test_loader=val_dl)

ans_df=pd.DataFrame(test_x["Id"])
y_comp_pred=[]
for i in range(len(y_pred)):
    y_comp_pred.append(-1)
    
for i in range(len(y_pred)):
    y_comp_pred[i]=int(y_pred[i])
    
ans_df["Genre"]=y_comp_pred
ans_df.to_csv("non_comp_test_pred_y", index=False)







