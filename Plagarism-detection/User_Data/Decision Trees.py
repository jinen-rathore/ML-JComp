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


train_data


# In[5]:


train_data.info()


# In[6]:


train_data.describe().transpose()


# In[7]:


refined_train_data = train_data.drop(["CustomerID", "City", "Latitude", "Longitude"], axis = 1)


# In[8]:


refined_test_data = test_data.drop(["CustomerID", "City", "Latitude", "Longitude"], axis = 1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


# In[10]:


le = LabelEncoder()
oe = OrdinalEncoder()


# In[11]:


refined_train_data[["Married", "Contract", "Payment Method", "Gender", "Offer", "Phone Service", "Internet Service"]] = oe.fit_transform(refined_train_data[["Married", "Contract", "Payment Method", "Gender", "Offer", "Phone Service", "Internet Service"]])


# In[12]:


refined_train_data["CustomerStatus"] = le.fit_transform(refined_train_data["CustomerStatus"])


# In[13]:


refined_test_data[["Married", "Contract", "Payment Method", "Gender", "Offer", "Phone Service", "Internet Service"]] = oe.fit_transform(refined_test_data[["Married", "Contract", "Payment Method", "Gender", "Offer", "Phone Service", "Internet Service"]])


# In[14]:


refined_train_data


# In[15]:


refined_test_data


# In[16]:


refined_train_data.isnull().sum()


# In[17]:


refined_test_data.isnull().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


from sklearn.impute import SimpleImputer


# In[19]:


si = SimpleImputer(missing_values = np.nan, strategy ='mean')


# In[20]:


refined_train_data["Avg Monthly Long Distance Charges"] = si.fit_transform(refined_train_data["Avg Monthly Long Distance Charges"].values.reshape(-1, 1))
refined_train_data["Avg Monthly GB Download"] = si.fit_transform(refined_train_data["Avg Monthly GB Download"].values.reshape(-1, 1)) 


# In[21]:


refined_test_data["Avg Monthly Long Distance Charges"] = si.fit_transform(refined_test_data["Avg Monthly Long Distance Charges"].values.reshape(-1, 1))
refined_test_data["Avg Monthly GB Download"] = si.fit_transform(refined_test_data["Avg Monthly GB Download"].values.reshape(-1, 1)) 


# In[22]:


refined_train_data.isnull().sum()


# In[23]:


refined_test_data.isnull().sum()


# In[24]:


from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


# In[25]:


X = refined_train_data.drop("CustomerStatus", axis = 1)
y = refined_train_data["CustomerStatus"]


# In[26]:


std_slc = StandardScaler()
pca = decomposition.PCA()
dec_tree = tree.DecisionTreeClassifier()


# In[27]:


pipe = Pipeline(steps=[('std_slc', std_slc),
                        ('pca', pca),
                        ('dec_tree', dec_tree)])


# In[28]:


n_components = list(range(1,X.shape[1]+1,1))


# In[29]:


criterion = ['gini', 'entropy']
max_depth = [2,4,6,8,10,12]


# In[30]:


parameters = dict(pca__n_components=n_components,
                    dec_tree__criterion=criterion,
                    dec_tree__max_depth=max_depth)


# In[31]:


clf_GS = GridSearchCV(pipe, parameters)
clf_GS.fit(X, y)


# In[32]:


print('Best Criterion:', clf_GS.best_estimator_.get_params()['dec_tree__criterion'])
print('Best max_depth:', clf_GS.best_estimator_.get_params()['dec_tree__max_depth'])
print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
print(); print(clf_GS.best_estimator_.get_params()['dec_tree'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[33]:


from sklearn.preprocessing import StandardScaler


# In[34]:


sc = StandardScaler()


# In[35]:


X = sc.fit_transform(X)


# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[38]:


from sklearn.tree import DecisionTreeClassifier


# In[39]:


dtree = DecisionTreeClassifier(criterion='entropy', max_depth=6)


# In[40]:


dtree.fit(X_train, y_train)


# In[41]:


predictions = dtree.predict(X_test)


# In[42]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[43]:


print(classification_report(y_test, predictions))


# In[44]:


print(confusion_matrix(y_test, predictions))


# In[45]:


print(accuracy_score(y_test, predictions))


# In[46]:


testData = sc.fit_transform(refined_test_data)


# In[47]:


predictions = dtree.predict(testData)
predictions


# In[48]:


predictions_test = le.inverse_transform(predictions)


# In[49]:


predictions_test


# In[50]:


df = pd.DataFrame({"CustomerStatus":predictions_test})
ID = test_data["CustomerID"]


# In[51]:


result = pd.concat([ID,df], axis=1, join='inner')
display(result)


# In[52]:


result.to_csv('results.csv', index=False)


# In[ ]:





# In[53]:


from sklearn import tree


# In[54]:


text_representation = tree.export_text(dtree)
print(text_representation)


# In[55]:


features = list(refined_train_data.columns[1:])
features


# In[58]:


fig = plt.figure(figsize=(30,30))
_ = tree.plot_tree(dtree, 
                   feature_names=features,
                   filled=True)

