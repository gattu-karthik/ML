#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import string
import  numpy  as  np
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import xgboost


# In[4]:


from lightgbm import LGBMClassifier


# In[13]:


# pip install lightgbm


# In[6]:


# train_data_path =
# validation_data_path =
# test_data_path =
# output_folder_path =


train_data_path = "COL774_drug_review"
test_data_path = "COL774_drug_review"
validation_data_path = "COL774_drug_review"
# output_folder_path = 


# In[7]:


train_data = pd.read_csv(train_data_path + "/DrugsComTrain.csv")
test_data = pd.read_csv(test_data_path + "/DrugsComTest.csv")
validation_data = pd.read_csv(validation_data_path + "/DrugsComVal.csv")


# In[5]:


# train_data


# In[8]:


def cleaned_data(message):
    stops = stopwords.words("english")
    words = message.split()
    final_mssg = []
    for word in words:
        if word not in stops:
            final_mssg.append(word.lower())
    final_mssg = " ".join(final_mssg)
    while "  " in final_mssg:
        final_mssg = final_mssg.replace("  ", " ")
    final_mssg  = final_mssg .translate(str.maketrans("", "", string.punctuation))
    final_mssg = "".join([i for i in final_mssg if not i.isdigit()])
    
    return final_mssg


# In[9]:


def get_sparse_features_and_vecs(data):

  reviews = data['review'].tolist()
  conditions = data['condition'].tolist()

  cleaned_reviews = [cleaned_data(rev) for rev in reviews]
  cleaned_conditions = [str(con).lower() for con in conditions]

  vectorizer_1 = TfidfVectorizer(lowercase=True,max_df=0.9,min_df=3,ngram_range = (1,3),stop_words = "english")
  fitted_vectorizer_rev = vectorizer_1.fit(cleaned_reviews)
  reviews_vecs = fitted_vectorizer_rev.transform(cleaned_reviews)

  vectorizer_2 = TfidfVectorizer()
  fitted_vectorizer_con = vectorizer_2.fit(cleaned_conditions)
  conditions_vecs = fitted_vectorizer_con.transform(cleaned_conditions)

  processed_data  = hstack([reviews_vecs,conditions_vecs])


  dates = data['date'].tolist()

  years = []
  months = []
  days = []
  mnt = {"January":1, "February":2, "March":3, "April":4, "May":5, "June":6,
    "July":7, "August":8, "September":9, "October":10, "November":11, "December":12}
  for date in dates :
      t = date.split()
      years.append(int(t[2]))
      days.append(int(t[1][:-1]))
      months.append(mnt[t[0]])


  df1 = pd.DataFrame(years)
  df2 = pd.DataFrame(months)
  df3 = pd.DataFrame(days)
  df4 = pd.DataFrame(data['usefulCount'].tolist())

  processed_dates = pd.concat([df1,df2,df3,df4],axis=1)

  sp_arr = csr_matrix(processed_dates)
  sdf = pd.DataFrame.sparse.from_spmatrix(sp_arr)
  sparse_dates = sdf.sparse.to_coo()

  # sparse_dates = csr_matrix(processed_dates.astype(pd.SparseDtype("float64",0)).sparse.to_coo())
  sparse_total_data = hstack([processed_data,sparse_dates])
  Y = data['rating'].tolist()
  Y = pd.DataFrame(Y)
  X = sparse_total_data

  return X,Y,fitted_vectorizer_rev,fitted_vectorizer_con


# In[10]:


def get_sparse_features(data,vec_1,vec_2):
  reviews = data['review'].tolist()
  conditions = data['condition'].tolist()

  cleaned_reviews = [cleaned_data(rev) for rev in reviews]
  cleaned_conditions = [str(con).lower() for con in conditions]

  reviews_vecs = vec_1.transform(cleaned_reviews)
  conditions_vecs = vec_2.transform(cleaned_conditions)

  processed_data  = hstack([reviews_vecs,conditions_vecs])

  dates = data['date'].tolist()

  years = []
  months = []
  days = []
  mnt = {"January":1, "February":2, "March":3, "April":4, "May":5, "June":6,
    "July":7, "August":8, "September":9, "October":10, "November":11, "December":12}
  for date in dates :
      t = date.split()
      years.append(int(t[2]))
      days.append(int(t[1][:-1]))
      months.append(mnt[t[0]])


  df1 = pd.DataFrame(years)
  df2 = pd.DataFrame(months)
  df3 = pd.DataFrame(days)
  df4 = pd.DataFrame(data['usefulCount'].tolist())

  processed_dates = pd.concat([df1,df2,df3,df4],axis=1)

  sp_arr = csr_matrix(processed_dates)
  sdf = pd.DataFrame.sparse.from_spmatrix(sp_arr)
  sparse_dates = sdf.sparse.to_coo()

  # sparse_dates = csr_matrix(processed_dates.astype(pd.SparseDtype("float64",0)).sparse.to_coo())
  sparse_total_data = hstack([processed_data,sparse_dates])
  Y = data['rating'].tolist()
  Y = pd.DataFrame(Y)
  X = sparse_total_data

  return X,Y


# In[11]:


X_train,Y_train, fitted_vectorizer_rev,fitted_vectorizer_con = get_sparse_features_and_vecs(train_data)
X_test , Y_test = get_sparse_features(test_data,fitted_vectorizer_rev,fitted_vectorizer_con)
X_validation,Y_validation = get_sparse_features(validation_data,fitted_vectorizer_rev,fitted_vectorizer_con)


# In[9]:


def part_a():
  clf = DecisionTreeClassifier()
  clf.fit(X_train,Y_train)

  train_acc = clf.score(X_train,Y_train)
  test_acc = clf.score(X_test,Y_test)
  validation_acc = clf.score(X_validation,Y_validation)

  print(100*train_acc)
  print(100*test_acc)
  print(100*validation_acc)


# In[ ]:


part_a()


# In[ ]:


# 100.0
# 57.61447755086858
# 58.12478042530327


# In[10]:


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


# In[ ]:


part_b()


# In[ ]:


# 32.97463421546746
# 32.901833872707655
# 33.04883341255244
# {'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 2}


# In[10]:


def part_c():
    clf = DecisionTreeClassifier(random_state=0)
    values = clf.cost_complexity_pruning_path(X_train, Y_train)

    ccp_alphas = values.ccp_alphas
    impurities = values.impurities

    ccp_alphas = ccp_alphas[:-1]
    impurities = impurities[:-1]
    print(len(ccp_alphas))
    models = []
    for ccp_alpha in ccp_alphas :
        print("----####----")
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
    plt.savefig("dt2_part_c_1.png")
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
    plt.savefig("dt2_part_c_2.png")
    plt.show()

    # ---------------Max_height VS ccp_alphas
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Max_height VS ccp_alphas")
    ax.set_ylabel("Max_height")
    ax.set_xlabel("ccp_alphas")
    ax.plot(ccp_alphas, height_of_trees, marker="o", drawstyle="steps-post")
    plt.legend()
    plt.savefig("dt2_part_c_3.png")
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
    plt.savefig("dt2_part_c_4.png")
    plt.show()

    opt_clf = DecisionTreeClassifier(random_state=0, ccp_alpha=0.012)
    opt_clf.fit(X_train,Y_train)
    train_acc = opt_clf.score(X_train,Y_train)
    test_acc = opt_clf.score(X_test,Y_test)
    validation_acc = opt_clf.score(X_validation,Y_validation)

    print(100*train_acc)
    print(100*test_acc)
    print(100*validation_acc)

    


# In[11]:


part_c()


# In[17]:


def part_d():
    clf = RandomForestClassifier(bootstrap=True,oob_score=True,verbose=0)

    parameters = {"n_estimators"      : [50,150,250,350,450],
                   "max_features"      : [0.4,0.6,0.8],
                   "min_samples_split"  : [2,5,8,10] }

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


# In[18]:


part_d()


# In[15]:


def part_e():
    clf = xgboost.XGBClassifier()
    valdtn_set = [(X_validation, Y_validation)]
#     clf.fit(X_train,Y_train,eval_set=valdtn_set,verbose=False)
    clf.fit(X_train,Y_train,verbose=False)

    parameters = {"n_estimators"      : [50,150,250,350,450],
                       "max_features"      : [0.4,0.6,0.8],
                       "max_depth"  : [40,50,60,70] }

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


# In[16]:


part_e()


# In[12]:


def part_f():
    clf = LGBMClassifier(boosting_type = 'goss')
    st = time.time()
    clf.fit(X_train,Y_train)
    et=time.time()
    train_acc = clf.score(X_train,Y_train)
    test_acc = clf.score(X_test,Y_test)
    validation_acc = clf.score(X_validation,Y_validation)

    print(100*train_acc)
    print(100*test_acc)
    print(100*validation_acc)
    print(et-st)


# In[ ]:


part_f()


# In[1]:


def part_g():
    for i in range(1,9):
        train_data = pd.read_csv(train_data_path + "/DrugsComTrain.csv")
        
        train_data = train_data.sample(n=i*2000)
        
        part_a()
        part_b()
        part_c()
        part_d()
        part_e()
        part_f()
        


# In[ ]:


part_g()

