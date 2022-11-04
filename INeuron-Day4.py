#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
df = pd.read_csv("D:/Project/Ineuron/iris.csv")
df.head()


# In[30]:


X = df[['sepal_length','petal_length','sepal_width','petal_width']]
Y = df['variety']


# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)
X_train


# In[34]:


X_test


# In[35]:


Y_train


# In[36]:


Y_test


# In[38]:


# check if the train and test data is 20% and 80% resp and same applies tp Y data.
X_train.count()
X_test.count()


# In[39]:


import pandas as pd
df = pd.read_csv("D:/Project/Ineuron/table2.csv")
df.head()


# In[40]:


X = df[['distance', 'years']]
y = df['price']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




