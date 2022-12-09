# #!/usr/bin/env python
# # coding: utf-8

# # In[19]:



# import sys
# sys.path.append('../Qa/qa')


# # In[22]:


# def random_prediction():
#     tp=0
#     tn=0
#     for i in range(len(test_data_pos)):
#         rand_val = np.random.random()
#         if (rand_val >= 0.5): tp += 1
       
#     for i in range(len(test_data_neg)) :
#         rand_val = np.random.random()
#         if (rand_val < 0.5): tn += 1

#     return tp,tn


# # In[26]:


# tp,tn = random_prediction()
# fn = len(test_data_pos) - tp
# fp = len(test_data_neg) - tn
# conf_mat = np.array([[tn,fn],[fp,tp]])
# print(tp)
# print(tn)
# print(conf_mat)
# print((tp+tn)/(len(test_data_pos)+len(test_data_neg)))


# # In[27]:


# majority_pred_accuracy = (max(len(test_data_neg), len(test_data_pos))/(len(test_data_pos)+len(test_data_neg)))*100
# print(majority_pred_accuracy)
# conf_mat = np.array([[0,0],[5000,10000]])
# print(conf_mat)

print("already doone in part 1")