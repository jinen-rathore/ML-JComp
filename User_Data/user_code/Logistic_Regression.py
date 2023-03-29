#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_data = pd.read_csv("train.csv")


# In[3]:


test_data = pd.read_csv("test.csv")


# In[4]:


train_data.head()


# In[5]:


train_data.describe().transpose()


# In[6]:


type(train_data["totalSales"])


# In[7]:


class_sales = []

for sale in train_data["totalSales"]:
    if sale <= 5:
        class_sales.append(1)
    elif sale > 5 and sale <= 10:
        class_sales.append(2)
    else:
        class_sales.append(3)
        
print(pd.Series(class_sales))


# In[8]:


train_data["Class"] = class_sales


# In[9]:


train_data.head()


# In[13]:


sns.pairplot(train_data.drop(["ID", "totalSales"], axis = 1), hue = "Class", palette = "bwr")


# In[52]:


cor = train_data.corr()

plt.figure(figsize = (12, 6))
sns.heatmap(cor, annot = True)


# Doing Label Encoding for train data and test data

# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[11]:


label_encoder = LabelEncoder()


# In[12]:


# Label Encoding the values of Location, Urban and US for train data
train_data["Location"] = label_encoder.fit_transform(train_data["Location"])
train_data["Urban"] = label_encoder.fit_transform(train_data["Urban"])
train_data["US"] = label_encoder.fit_transform(train_data["US"])


# In[13]:


train_data["Location"].unique()


# In[14]:


train_data["Urban"].unique()


# In[15]:


train_data["US"].unique()


# In[16]:


train_data.head()


# In[17]:


# Label Encoding the values of Location, Urban and US for test data
test_data["Location"] = label_encoder.fit_transform(test_data["Location"])
test_data["Urban"] = label_encoder.fit_transform(test_data["Urban"])
test_data["US"] = label_encoder.fit_transform(test_data["US"])


# In[18]:


test_data["Location"].unique()


# In[19]:


test_data["Urban"].unique()


# In[20]:


test_data["US"].unique()


# Training the model using train, test split

# In[21]:


from sklearn.model_selection import train_test_split


# In[23]:


X = train_data.drop(["ID", "Class"],axis = 1)
y = train_data["Class"]


# In[24]:


X


# In[25]:


y


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[27]:


from sklearn.linear_model import LogisticRegression


# In[29]:


logmodel = LogisticRegression(max_iter=1e8)
logmodel.fit(X_train,y_train)


# In[30]:


predictions = logmodel.predict(X_test)


# In[31]:


from sklearn.metrics import classification_report


# In[32]:


print(classification_report(y_test,predictions))


# In[33]:


from sklearn.metrics import accuracy_score


# In[34]:


print(accuracy_score(y_test, predictions) * 100) 


# Predictions on the test data

# In[38]:


preds = logmodel.predict(test_data.drop("ID", axis = 1))


# In[43]:


preds


# In[48]:


df = pd.DataFrame({"Category":preds})
ID = test_data["ID"]


# In[49]:


result = pd.concat([ID,df], axis=1, join='inner')
display(result)


# In[50]:


result.to_csv('results.csv', index=False)


# In[ ]:




