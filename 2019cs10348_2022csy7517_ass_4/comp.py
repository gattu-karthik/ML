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


# In[9]:


from PIL import Image

# In[ ]:


root_path  =  sys.argv[1]


# In[5]:


train_x_path = root_path+"/train_x.csv"
test_x_path = root_path+"/non_comp_test_x.csv"
train_y_path = root_path+"/train_y.csv"
test_y_path = root_path+"/non_comp_test_y.csv"
images_path = root_path+"/images/images/"
comp_test_path = root_path + "/comp_test_x.csv"


# In[6]:


train_x=pd.read_csv(train_x_path)
test_x=pd.read_csv(test_x_path)
train_y=pd.read_csv(train_y_path)
test_y=pd.read_csv(test_y_path)
comp_test_x=pd.read_csv(comp_test_path)


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
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




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



# In[90]:


device = "cpu"
if (torch.cuda.is_available()):
    # if GPU available, use cuda (on a cpu, training will take a considerable length of time!)
    device = "cuda"







import torch
from transformers import XLMTokenizer, XLMWithLMHeadModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import os
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

import pandas as pd
import numpy as np
import os
import gc

import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# set a seed value
torch.manual_seed(555)


MODEL_TYPE = 'xlm-roberta-base'

#tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_TYPE)
model = XLMWithLMHeadModel.from_pretrained(MODEL_TYPE)

L_RATE = 1e-5
MAX_LEN = 256

NUM_EPOCHS = 3
BATCH_SIZE = 32
NUM_CORES = os.cpu_count()

tokenizer =AutoTokenizer.from_pretrained(MODEL_TYPE)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

tokenizer.vocab_size

sentence1 = 'Hello there.'

encoded_dict = tokenizer.encode_plus(
            sentence1,                
            add_special_tokens = True,
            max_length = MAX_LEN,     
            pad_to_max_length = True,
            return_attention_mask = True,  
            return_tensors = 'pt' # return pytorch tensors
       )


encoded_dict

input_ids = encoded_dict['input_ids'][0]
att_mask = encoded_dict['attention_mask'][0]

print(input_ids)
print(att_mask)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = df['Genre']
#         self.texts = [tokenizer(text, 
#                                padding='max_length', max_length = 512, truncation=True,
#                                 return_tensors="pt") for text in df['text']]
        df['Title'] = df['Title'].fillna("")
        #df['hypothesis'] = df['hypothesis'].fillna("")
        self.title = df['Title']
        #self.hypothesis = df['hypothesis']
        
        self.texts = []

        for i in range(len(df['Title'])):
#             print(df['premise'].iloc[i] , df['hypothesis'].iloc[i])
            text = " " + df['Title'].iloc[i] + "  " 
#             text = self.premise.iloc[i]
#             print(text)
#             break
            d = tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
#             print(d)
            d['input_ids'] = d['input_ids'][0].to(device)
            #d['token_type_ids'] = d['token_type_ids'][0].to(device)
            d['attention_mask'] = d['attention_mask'][0].to(device)
            self.texts.append(d)
#         print(self.texts)
        
    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y



#df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), 
                                     #[int(.8*len(df)), int(.9*len(df))])

np.random.seed(112)


def mean_pooling(model_output, attention_mask):
    #print("Pooling")
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    #print(attention_mask.size())
    temp=attention_mask.unsqueeze(-1)
    #print(temp.size())
    input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



from transformers import XLMRobertaForSequenceClassification

model = XLMRobertaForSequenceClassification.from_pretrained(
    MODEL_TYPE,
    num_labels = 30, # The number of output labels. 2 for binary classification.
)


class XLMClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(XLMClassifier, self).__init__()

        self.xlm = model
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 30)
#         self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, input_id, mask):
        #print("Forward")
        #print(input_id)
        pooled_output = self.xlm(**input_id)
        #print(pooled_output[0].size())
        #print(len(pooled_output))
        #output = mean_pooling(pooled_output, input_id['attention_mask'])
        #print(output.size())
        #dropout_output = self.dropout(output)
        #print(dropout_output.size())
        #linear_output = self.linear(dropout_output)
        #print(linear_output.size())
        #final_layer = self.softmax(linear_output)
        pooled_output1=pooled_output[0]
        final_layer=self.softmax(pooled_output1)
        #print(final_layer.size())

        return final_layer

import copy

def train(model, train_data, val_data, learning_rate, epochs):

    train = Dataset(train_data)
    val= Dataset(val_data)
    print("here1")
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=5, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=5)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)
    print("here2")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0
            cnt=0
            for train_input, train_label in tqdm(train_dataloader):
                cnt+=1
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
#                 input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(train_input, mask)
                
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc
                if(cnt%1000==0):
                    print("batch_loss:")
                    print(batch_loss)
                    print("acc:")
                    print(acc)
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
#                     input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(val_input, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
                if total_acc_val > best_acc:
                    best_acc = total_acc_val
                    best_model_wts = copy.deepcopy(model.state_dict())
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')
            
            model.load_state_dict(best_model_wts)
    
    torch.save(model.state_dict(), "xlm_model-v3")


EPOCHS = 50
model1 = XLMClassifier()
LR = 1e-6

train_x1=pd.DataFrame( train_x)

train_x1['Genre']=dataset_y


val_x1=pd.DataFrame( test_x)

val_x1['Genre']=test_y['Genre']

test_x1=pd.DataFrame( comp_test_x)

tokenizer =AutoTokenizer.from_pretrained(MODEL_TYPE)

train(model1, train_x1, val_x1, LR, EPOCHS)



class Dataset1(torch.utils.data.Dataset):

    def __init__(self, df):

        #self.labels = df['Genre']
#         self.texts = [tokenizer(text, 
#                                padding='max_length', max_length = 512, truncation=True,
#                                 return_tensors="pt") for text in df['text']]
        df['Title'] = df['Title'].fillna("")
        #df['hypothesis'] = df['hypothesis'].fillna("")
        self.title = df['Title']
        #self.hypothesis = df['hypothesis']
        
        self.texts = []

        for i in range(len(df['Title'])):
#             print(df['premise'].iloc[i] , df['hypothesis'].iloc[i])
            text = " " + df['Title'].iloc[i] + "  " 
#             text = self.premise.iloc[i]
#             print(text)
#             break
            d = tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
#             print(d)
            d['input_ids'] = d['input_ids'][0].to(device)
            #d['token_type_ids'] = d['token_type_ids'][0].to(device)
            d['attention_mask'] = d['attention_mask'][0].to(device)
            self.texts.append(d)
#         print(self.texts)
        
    #def classes(self):
        #return self.labels

    def __len__(self):
        return len(self.texts)

    #def get_batch_labels(self, idx):
        # Fetch a batch of labels
        #return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        #batch_y = self.get_batch_labels(idx)

        return batch_texts



#df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), 
                                     #[int(.8*len(df)), int(.9*len(df))])

np.random.seed(112)


def test(model, test_data ):

            test = Dataset1(test_data)

            test_dataloader = torch.utils.data.DataLoader(test, batch_size=5, shuffle=False)
            #val_dataloader = torch.utils.data.DataLoader(val, batch_size=5)


            print("here2")
    
            if use_cuda:

                model = model.cuda()


            with torch.no_grad():
                y_pred=[]

                for test_input in test_dataloader:

                    #val_label = val_label.to(device)
                    mask = test_input['attention_mask'].to(device)
#                     input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(test_input, mask)

                    #batch_loss = criterion(output, val_label)
                    #total_loss_val += batch_loss.item()
                    
                    #acc = (output.argmax(dim=1) == val_label).sum().item()
                    #total_acc_val += acc
                    #print(output)
                    for o in range(len(output)):
                        y_pred.append(output[o])
            return y_pred
           




y_preds=test(model1, test_x1)

y_comp_pred=[]
for i in range(len(y_preds)):
    y_comp_pred.append(-1)
    
for i in range(len(y_preds)):
    predicted = torch.argmax(y_preds[i])
    y_comp_pred[i]=int(predicted)

ans_df=pd.DataFrame(test_x["Id"])

    
ans_df["Genre"]=y_comp_pred
ans_df.to_csv("comp_test_y", index=False)







