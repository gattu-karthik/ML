#!/usr/bin/env python
# coding: utf-8

# In[533]:


import numpy as np
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import xgboost


# In[534]:


# train_data_path =
# validation_data_path =
# test_data_path =
# output_folder_path =


train_data_path = "COL774_mammography"
test_data_path = "COL774_mammography"
validation_data_path = "COL774_mammography"
# output_folder_path = 


# In[535]:


train_data = pd.read_csv(train_data_path + "/train.csv")
test_data = pd.read_csv(test_data_path + "/test.csv")
validation_data = pd.read_csv(validation_data_path + "/val.csv")


# In[510]:


train_data = train_data.iloc[:,1:6]
test_data = test_data.iloc[:,1:6]
validation_data = validation_data.iloc[:,1:6]


# In[511]:


# -----------data drooping
train_data.drop(train_data[(train_data['Age'] == "?") | (train_data['Shape'] == "?") | (train_data['Margin'] == "?") | (train_data['Density'] == "?") | (train_data['Severity'] == "?")   ].index, inplace=True)
test_data.drop(test_data[(test_data['Age'] == "?") | (test_data['Shape'] == "?") | (test_data['Margin'] == "?") | (test_data['Density'] == "?") | (test_data['Severity'] == "?")   ].index, inplace=True)
validation_data.drop(validation_data[(validation_data['Age'] == "?") | (validation_data['Shape'] == "?") | (validation_data['Margin'] == "?") | (validation_data['Density'] == "?") | (validation_data['Severity'] == "?")   ].index, inplace=True)


# In[512]:


Y_train = train_data.iloc[:,4:5].to_numpy()
X_train = train_data.iloc[:,0:4].to_numpy()

Y_test = test_data.iloc[:,4:5].to_numpy()
X_test = test_data.iloc[:,0:4].to_numpy()

Y_validation = validation_data.iloc[:,4:5].to_numpy()
X_validation = validation_data.iloc[:,0:4].to_numpy()


# In[513]:


# print(X_train.shape)
# print(Y_train.shape)


# In[540]:


def part_a():
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train,Y_train)

    train_acc = clf.score(X_train,Y_train)
    test_acc = clf.score(X_test,Y_test)
    validation_acc = clf.score(X_validation,Y_validation)

    print(100*train_acc)
    print(100*test_acc)
    print(100*validation_acc)


    fig=(plt.subplots(figsize = (40,40)))[0]
    ftrs=['Age','Shape','Margin','Density']
    cnms = ['0','1']
    tree.plot_tree(clf,feature_names = ftrs,class_names=cnms,filled = True)
    fig.savefig('e_a_mode.png')


# In[515]:


part_a()


# In[542]:


def part_b():
    clf = DecisionTreeClassifier(random_state=0)

    parameters = {'max_depth':[1,2,3,4,5], 
                  'min_samples_leaf':[1,2,3,4,5], 
                  'min_samples_split':[2,3,4,5] }

    grid_obj = GridSearchCV(clf, parameters, cv = 5 ,scoring='accuracy',verbose=0)

    combos = grid_obj.fit(X_train,Y_train)

    opt_clf = combos.best_estimator_

    opt_clf.fit(X_train,Y_train)

    train_acc = opt_clf.score(X_train,Y_train)
    test_acc = opt_clf.score(X_test,Y_test)
    validation_acc = opt_clf.score(X_validation,Y_validation)

    print(100*train_acc)
    print(100*test_acc)
    print(100*validation_acc)

    print(combos.best_params_)

    ftrs=['Age','Shape','Margin','Density']
    cnms = ['0','1']
    fig=(plt.subplots(figsize = (40,40)))[0]
    tree.plot_tree(opt_clf,feature_names = ftrs,class_names=cnms,filled = True)
    fig.savefig('e_b_mode.png')


# In[517]:


part_b()


# In[544]:


def part_c():
    clf = DecisionTreeClassifier(random_state=0)
    values = clf.cost_complexity_pruning_path(X_train, Y_train)

    ccp_alphas = values.ccp_alphas
    impurities = values.impurities

    ccp_alphas = ccp_alphas[:-1]
    impurities = impurities[:-1]

    models = []
    for ccp_alpha in ccp_alphas :
        t = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        t.fit(X_train, Y_train)
        models.append(t)


    train_accs = [model.score(X_train, Y_train) for model in models]
    test_accs = [model.score(X_test, Y_test) for model in models]
    validation_accs = [model.score(X_validation, Y_validation) for model in models]

    no_of_nodes = [model.tree_.node_count for model in models]
    height_of_trees = [model.tree_.max_depth for model in models]

    # ------impurites vs alphasb
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Impurity of leaves VS ccp_alphas")
    ax.set_ylabel("Impurity of leaves")
    ax.set_xlabel("ccp_alphas")
    ax.plot(ccp_alphas, impurities, marker="o", drawstyle="steps-post")
    plt.legend()
    plt.savefig("e_part_c_1_mode.png")
    plt.show()
    # plt.close()

    # ---------------No_of_nodes VS ccp_alphas
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("No_of_nodes VS ccp_alphas")
    ax.set_ylabel("No_of_nodes")
    ax.set_xlabel("ccp_alphas")
    ax.plot(ccp_alphas, no_of_nodes, marker="o", drawstyle="steps-post")
    plt.legend()
    plt.savefig("e_part_c_2_mode.png")
    plt.show()

    # ---------------Max_height VS ccp_alphas
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Max_height VS ccp_alphas")
    ax.set_ylabel("Max_height")
    ax.set_xlabel("ccp_alphas")
    ax.plot(ccp_alphas, height_of_trees, marker="o", drawstyle="steps-post")
    plt.legend()
    plt.savefig("e_part_c_3_mode.png")
    plt.show()

    # -------------Training,test,val --_accuracy  vs ccp_alpha
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Training_accuracy VS ccp_alphas")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("ccp_alphas")
    ax.plot(ccp_alphas, train_accs, marker="o",label='train', drawstyle="steps-post")
    ax.plot(ccp_alphas, test_accs, marker="o",label='test', drawstyle="steps-post")
    ax.plot(ccp_alphas, validation_accs, marker="o",label='validation', drawstyle="steps-post")
    plt.legend()
    plt.savefig("e_part_c_4_mode.png")
    plt.show()

    opt_clf = DecisionTreeClassifier(random_state=0, ccp_alpha=0.012)
    opt_clf.fit(X_train,Y_train)
    train_acc = opt_clf.score(X_train,Y_train)
    test_acc = opt_clf.score(X_test,Y_test)
    validation_acc = opt_clf.score(X_validation,Y_validation)

    print(100*train_acc)
    print(100*test_acc)
    print(100*validation_acc)

    ftrs=['Age','Shape','Margin','Density']
    cnms = ['0','1']
    fig=(plt.subplots(figsize = (15,5)))[0]
    tree.plot_tree(opt_clf,feature_names = ftrs,class_names=cnms,filled = True)
    fig.savefig('e_part_c_5_mode.png')


# In[519]:


part_c()


# In[471]:


def part_d():
    clf = RandomForestClassifier(bootstrap=True,oob_score=True,verbose=0)

    parameters = {"n_estimators"      : [100,200,300,400,500],
                   "max_features"      : ['sqrt'],
                   "min_samples_split"  : [2,3,4,5] }

    grid_obj = GridSearchCV(clf, parameters, cv = 5 ,scoring='accuracy',verbose=0)
    combos = grid_obj.fit(X_train,Y_train)

    opt_clf = combos.best_estimator_
    opt_clf.fit(X_train,Y_train)

    train_acc = opt_clf.score(X_train,Y_train)
    test_acc = opt_clf.score(X_test,Y_test)
    validation_acc = opt_clf.score(X_validation,Y_validation)
    oob_acc = opt_clf.oob_score_

    print(100*train_acc)
    print(100*test_acc)
    print(100*validation_acc)
    print(100*oob_acc)

    print(combos.best_params_)


# In[419]:


part_d()


# In[522]:


train_data = pd.read_csv(train_data_path + "/train.csv")
test_data = pd.read_csv(test_data_path + "/test.csv")
validation_data = pd.read_csv(validation_data_path + "/val.csv")


# In[536]:


train_data = train_data.iloc[:,1:6]
test_data = test_data.iloc[:,1:6]
validation_data = validation_data.iloc[:,1:6]


# In[537]:


train_data.replace('?', 'NaN', inplace=True)
test_data.replace('?', 'NaN', inplace=True)
validation_data.replace('?', 'NaN', inplace=True)


# In[538]:


train_data[train_data.columns] = train_data[train_data.columns].apply(pd.to_numeric, errors='coerce')
test_data[test_data.columns] = test_data[test_data.columns].apply(pd.to_numeric, errors='coerce')
validation_data[validation_data.columns] = validation_data[validation_data.columns].apply(pd.to_numeric, errors='coerce')


# In[526]:


# -------fillling with median
train_data = train_data.fillna(train_data.median())
test_data = test_data.fillna(test_data.median())
validation_data = validation_data.fillna(validation_data.median())


# In[528]:


part_a()


# In[530]:


part_b()


# In[532]:


part_c()


# In[465]:


part_d()


# In[539]:


# -------fillling with mode
train_data['Age'].fillna(train_data['Age'].mode()[0],inplace=True)
train_data['Shape'].fillna(train_data['Shape'].mode()[0],inplace=True)
train_data['Margin'].fillna(train_data['Margin'].mode()[0],inplace=True)
train_data['Density'].fillna(train_data['Density'].mode()[0],inplace=True)
train_data['Severity'].fillna(train_data['Severity'].mode()[0],inplace=True)

test_data['Age'].fillna(test_data['Age'].mode()[0],inplace=True)
test_data['Shape'].fillna(test_data['Shape'].mode()[0],inplace=True)
test_data['Margin'].fillna(test_data['Margin'].mode()[0],inplace=True)
test_data['Density'].fillna(test_data['Density'].mode()[0],inplace=True)
test_data['Severity'].fillna(test_data['Severity'].mode()[0],inplace=True)

validation_data['Age'].fillna(validation_data['Age'].mode()[0],inplace=True)
validation_data['Shape'].fillna(validation_data['Shape'].mode()[0],inplace=True)
validation_data['Margin'].fillna(validation_data['Margin'].mode()[0],inplace=True)
validation_data['Density'].fillna(validation_data['Density'].mode()[0],inplace=True)
validation_data['Severity'].fillna(validation_data['Severity'].mode()[0],inplace=True)


# In[541]:


part_a()


# In[543]:


part_b()


# In[545]:


part_c()


# In[480]:


part_d()


# In[489]:


def part_f():
    clf = xgboost.XGBClassifier()
    valdtn_set = [(X_validation, Y_validation)]
    clf.fit(X_train,Y_train,eval_set=valdtn_set,verbose=False)

    parameters = {"n_estimators"      : [10,20,30,40,50],
                       "subsample"      : [0.1,0.2,0.3,0.4,0.5,0.6],
                       "max_depth"  : [4,5,6,7,8,9,10] }

    grid_obj = GridSearchCV(clf, parameters, cv = 5 ,scoring='accuracy',verbose=0)
    combos = grid_obj.fit(X_train,Y_train)

    opt_clf = combos.best_estimator_
    opt_clf.fit(X_train,Y_train)

    train_acc = opt_clf.score(X_train,Y_train)
    test_acc = opt_clf.score(X_test,Y_test)
    validation_acc = opt_clf.score(X_validation,Y_validation)

    print(100*train_acc)
    print(100*test_acc)
    print(100*validation_acc)


    print(combos.best_params_)


# In[506]:


part_f()


# In[ ]:




