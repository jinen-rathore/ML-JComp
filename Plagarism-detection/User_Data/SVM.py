#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train = pd.read_csv("wheet_train.csv")
test = pd.read_csv("wheet_test.csv")


# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


sns.pairplot(train, hue = "Type", palette = "Dark2")


# In[ ]:





# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X = train.drop(["ID", "Type"], axis = 1)
y = train["Type"]


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[9]:


from sklearn.svm import SVC


# In[10]:


svc_model = SVC()
# C: Penalty parameter of the error term high value: strict margin; low value: soft margin  
# gamma: how far the influence of a single training example reaches: Low: far; high: close


# In[11]:


svc_model.fit(X_train, y_train)


# In[12]:


preds = svc_model.predict(X_test)


# In[13]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[14]:


print(classification_report(y_test, preds))


# In[15]:


print(confusion_matrix(y_test, preds))


# In[16]:


print(accuracy_score(y_test, preds))


# In[ ]:





# Using Grid Search CV to furthur fine tune the parameters

# In[17]:


from sklearn.model_selection import GridSearchCV


# In[54]:


param_grid = {"C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01, 0.001], "kernel": ['rbf', 'poly', 'sigmoid', 'linear']}


# In[55]:


grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)


# In[56]:


grid.fit(X_train, y_train)


# In[57]:


grid.best_params_


# In[58]:


grid.best_estimator_


# In[59]:


grid_preds = grid.predict(X_test)


# In[60]:


print(confusion_matrix(y_test, grid_preds))


# In[61]:


print(accuracy_score(y_test, grid_preds))


# In[62]:


print(classification_report(y_test, grid_preds))


# Building the model

# In[63]:


grid.fit(X, y)


# In[64]:


grid.best_params_


# In[65]:


grid.best_estimator_


# In[66]:


final_grid_preds = grid.predict(test.drop("ID", axis = 1))


# In[67]:


final_grid_preds


# In[68]:


df = pd.DataFrame({"Type":final_grid_preds})
ID = test["ID"]


# In[69]:


result = pd.concat([ID,df], axis=1, join='inner')
display(result)


# In[70]:


result.to_csv('results.csv', index=False)

