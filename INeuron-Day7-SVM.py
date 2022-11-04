#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
iris


# In[33]:


df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()


# In[34]:


df['target'] = iris.target
df.head()


# In[35]:


from sklearn.model_selection import train_test_split
X = df.drop(['target'], axis='columns')
y = df.target


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=9)


# In[66]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)


# In[67]:


model.score(X_test,y_test)


# In[68]:


model.predict([[5.9, 3. , 5.1, 1.8]])


# In[69]:


X.shape


# In[ ]:





# In[ ]:




