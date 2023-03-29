#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[2]:


crimeTrain = pd.read_csv("crime_train.csv")
crimeTest = pd.read_csv("crime_test.csv")


# In[3]:


crimeTrain.head()


# In[4]:


crimeTest.head()


# In[5]:


crimeTrain.info()


# In[6]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
crimeTrain.describe().transpose()


# In[7]:


crimeTest.info()


# feature selection

# In[8]:


cor = crimeTrain.corr()

plt.figure(figsize = (40, 40))
sns.heatmap(cor, annot = True)


# In[ ]:





# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X = crimeTrain.drop(["ViolentCrimesPerPop", "ID"], axis = 1)
y = crimeTrain["ViolentCrimesPerPop"]


# In[ ]:





# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:





# In[12]:


lm = LinearRegression()
lm.fit(X_train, y_train)


# In[13]:


lm.intercept_


# In[14]:


lm.coef_


# In[15]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[16]:


preds = lm.predict(X_test)


# In[17]:


plt.scatter(y_test, preds)


# In[18]:


sns.distplot((y_test-preds),bins=50);


# In[19]:


from sklearn import metrics


# In[20]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, preds)))


# In[21]:


p = lm.predict(crimeTest.drop("ID", axis = 1))


# In[22]:


crime_df = pd.DataFrame({"Prediction":p})
ID = crimeTest["ID"]


# In[23]:


result = pd.concat([ID,crime_df], axis=1, join='inner')
display(result)


# In[24]:


result.to_csv('results.csv', index=False)


# In[25]:


plm = PolynomialFeatures(degree=3)


# In[26]:


p = plm.fit_transform(X_train)


# In[27]:


plm.fit(X_train, y_train)


# In[28]:


lm = LinearRegression()
lm.fit(p, y_train)


# In[29]:


lm.coef_


# In[30]:


preds = lm.predict(plm.fit_transform(X_test))


# In[31]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, preds)))

