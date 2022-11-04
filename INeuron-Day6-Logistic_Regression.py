#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
df = pd.read_csv("D:/Project/Ineuron/result.csv")
df.head(10)


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.results,train_size=0.8,random_state=10)


# In[10]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)


# In[11]:


y_predicted = model.predict(X_test)
y_predicted


# In[12]:


model.score(X_test,y_test)


# In[14]:


from sklearn import datasets
from sklearn import linear_model


# In[15]:


digits = datasets.load_digits()
digits.keys()


# In[16]:


digits.target_names


# In[17]:


digits.data.shape


# In[18]:


digits.images.shape


# In[21]:


digits.images[0]


# In[23]:


import pylab as py
py.matshow(digits.images[0])
py.gray()


# In[24]:


X = digits.data
y = digits.target


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)


# In[26]:


digreg = LogisticRegression()


# In[27]:


digreg.fit(X_train, y_train)


# In[30]:


y_pred = digreg.predict(X_test)


# In[34]:


from sklearn.metrics import accuracy_score


# In[38]:


print(accuracy_score(y_test, y_pred))


# In[40]:


y_pred


# In[ ]:





# In[ ]:





# In[ ]:




