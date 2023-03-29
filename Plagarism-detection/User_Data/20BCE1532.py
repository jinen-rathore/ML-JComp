#!/usr/bin/env python
# coding: utf-8

# ### Preprocessing

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# For Train Data

# In[3]:


data.head()


# In[4]:


data['Embarked'].unique()


# In[5]:


from sklearn.preprocessing import LabelEncoder


# In[6]:


label_encoder = LabelEncoder()


# In[7]:


data['Embarked'] = label_encoder.fit_transform(data['Embarked'])


# In[8]:


data['Embarked'].unique()


# In[9]:


data['Sex'] = label_encoder.fit_transform(data['Sex'])


# In[10]:


data['Sex'].unique()


# In[11]:


data.drop(['Name', 'Cabin', 'Ticket'], axis = 1, inplace = True)


# In[12]:


data.head()


# In[13]:


data.head()


# In[14]:


print(data.isnull().sum())


# In[15]:


data['Age'] = data['Age'].fillna(data['Age'].mean())


# In[16]:


print(data.isnull().sum())


# #### For Test Data

# In[17]:


test.head()


# In[18]:


test['Embarked'].unique()


# In[19]:


from sklearn.preprocessing import LabelEncoder


# In[20]:


label_encoder = LabelEncoder()


# In[21]:


test['Embarked'] = label_encoder.fit_transform(test['Embarked'])


# In[22]:


test['Embarked'].unique()


# In[23]:


test['Sex'] = label_encoder.fit_transform(test['Sex'])


# In[24]:


test['Sex'].unique()


# In[25]:


test.drop(['Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)


# In[26]:


test.head()


# In[27]:


print(data.isnull().sum())


# ### Model KNN From Scratch

# In[28]:


from scipy import stats

class KNN:
    
    def __init__(self, k=7):
        self.k = k
        self.X_train = None
        self.y_train = None
        
    @staticmethod
    def _euclidean_distance(p, q):
        return np.sqrt(np.sum((p - q) ** 2))
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        predictions = []
        for p in X:
            euc_distances = [self._euclidean_distance(p, q) for q in self.X_train]
            sorted_k = np.argsort(euc_distances)[:self.k]
            k_nearest = [self.y_train[y] for y in sorted_k]
            predictions.append(stats.mode(k_nearest)[0][0])
            
        return np.array(predictions)


# In[29]:


train_data = np.array(data.drop("Survived", axis = 1))
preds_data = np.array(data["Survived"])
predict_data = np.array(test)


# In[30]:


model = KNN()
model.fit(train_data, preds_data)
preds = model.predict(predict_data)


# In[31]:


survived_df = pd.DataFrame({"Survived":preds})
PassengerId_df = test["PassengerId"]


# In[32]:


PassengerId_df


# In[33]:


survived_df


# In[34]:


result = pd.concat([PassengerId_df,survived_df], axis=1, join='inner')
display(result)


# In[35]:


result.to_csv('results.csv', index=False)


# ### Ploting

# In[36]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(np.array(data.drop("Survived",axis = 1)), np.array(data["Survived"]), test_size=0.3, random_state=42)


# In[37]:


from sklearn.metrics import accuracy_score
evals = []

for k in range(1, 16, 2):
    model = KNN(k=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    evals.append({'k': k, 'accuracy': accuracy})


# In[38]:


import matplotlib.pyplot as plt

evals = pd.DataFrame(evals)
best_k = evals.sort_values(by='accuracy', ascending=False).iloc[0]

plt.figure(figsize=(16, 8))
plt.plot(evals['k'], evals['accuracy'], lw=3, c='#087E8B')
plt.scatter(best_k['k'], best_k['accuracy'], s=200, c='#087E8B')
plt.title(f"K Parameter Optimization, Optimal k = {int(best_k['k'])}", size=20)
plt.xlabel('K', size=14)
plt.ylabel('Accuracy', size=14)
plt.show()

