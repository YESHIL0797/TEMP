#!/usr/bin/env python
# coding: utf-8

# In[20]:


from pandas import read_csv
data = read_csv("D:/Project/Ineuron/purchase.csv")
data


# In[7]:


data.describe()


# In[21]:


#mean
data['Ball'] = data['Ball'].fillna(data['Ball'].mean())
data


# In[24]:


#Medina
data['Bat'] = data['Bat'].fillna(data['Bat'].median())
data


# In[25]:


# standard deviation
data['apple'] = data['apple'].fillna(data['apple'].std())
data


# In[26]:


# Min
data['Orange'] = data['Orange'].fillna(data['Orange'].min())
data


# In[27]:


# Max
data['Price'] = data['Price'].fillna(data['Price'].max())
data


# In[46]:


from numpy import set_printoptions
from sklearn import preprocessing


# In[48]:



name = ["A","B","C","D","E","F","G","H","I"]
a = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv',names=name)
a


# In[54]:


a.describe()


# In[51]:


scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
rescaled = scaler.fit_transform(a)
set_printoptions(precision=2)
rescaled


# In[53]:


from sklearn.preprocessing import StandardScaler
data_scaler = StandardScaler().fit(a)
data_rescaled = data_scaler.transform(a)
data_rescaled


# In[56]:


from sklearn.preprocessing import Binarizer
binary = Binarizer(threshold=0.5)
binary1 = binary.transform(a)
binary1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




