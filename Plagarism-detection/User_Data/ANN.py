#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf


# In[2]:


train_data = pd.read_csv("train.csv")


# In[3]:


test_data = pd.read_csv("test.csv")


# In[4]:


train_data.head()


# In[5]:


train_data.info()


# In[6]:


train_data.describe().transpose()


# In[7]:


train_data.isnull().sum()


# In[55]:


cor = train_data.corr()

plt.figure(figsize = (20, 25))
sns.heatmap(cor, annot = True)


# In[ ]:





# In[ ]:





# Encoding and scaling the data

# In[9]:


ref_train_data = train_data.drop(["ID"], axis = 1)


# In[10]:


ref_test_data = test_data.drop(["ID"], axis = 1)


# In[11]:


from sklearn.preprocessing import OrdinalEncoder


# In[12]:


oe = OrdinalEncoder()


# In[13]:


ref_train_data[['gender', 'oral', 'tartar']] = oe.fit_transform(ref_train_data[['gender', 'oral', 'tartar']])


# In[14]:


ref_test_data[['gender', 'oral', 'tartar']] = oe.fit_transform(ref_test_data[['gender', 'oral', 'tartar']])


# In[ ]:





# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X = ref_train_data.drop("smoking", axis = 1)
y = ref_train_data["smoking"]


# In[17]:


X


# In[ ]:





# In[ ]:





# In[18]:


from sklearn.preprocessing import StandardScaler


# In[19]:


sc = StandardScaler()


# In[20]:


X = sc.fit_transform(X)


# In[21]:


ref_test_data = sc.transform(ref_test_data)


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[42]:


#Initialising ANN
ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units = 100, input_dim = 25, kernel_initializer='uniform', activation="relu"))

ann.add(tf.keras.layers.Dense(units=100,kernel_initializer='uniform',activation="relu"))

ann.add(tf.keras.layers.Dense(units=20,kernel_initializer='uniform',activation="relu"))

#Adding Output Layer
ann.add(tf.keras.layers.Dense(units=1,kernel_initializer='uniform',activation="sigmoid"))
# ann.add(tf.keras.layers.Dense(units=1,kernel_initializer='uniform',activation="sigmoid"))

#Compiling ANN
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])


# In[43]:


#Fitting ANN
ann.fit(X_train,y_train,batch_size=32,epochs = 50)


# In[44]:


ann.predict(X_test)


# In[45]:


preds = ann.predict(X_test)
preds = (preds > 0.5)


# In[46]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[47]:


print(confusion_matrix(y_test,preds))


# In[48]:


print(accuracy_score(y_test, preds)*100)


# In[49]:


ann.fit(X, y, batch_size = 32, epochs=50)


# In[50]:


preds = ann.predict(ref_test_data)
preds = (preds > 0.5)
preds


# In[51]:


l = []
smoking = []
for i in preds:
    l.append(*i)
for i in l:
    if i == True:
        smoking.append(1)
    else:
        smoking.append(0)


# In[52]:


df = pd.DataFrame({"smoking":smoking})
ID = test_data["ID"]


# In[53]:


result = pd.concat([ID,df], axis=1, join='inner')
display(result)


# In[54]:


result.to_csv('results.csv', index=False)

