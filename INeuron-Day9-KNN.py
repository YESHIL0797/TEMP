#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os as os
os.getcwd()
os.chdir('D:/Project/Ineuron')
import pandas as pd
dataset = pd.read_csv("Car.csv")
dataset.head(10)


# In[21]:


from matplotlib import pyplot as plt
plt.scatter(dataset.Age,dataset['Income'])


# In[24]:


df1 = dataset[dataset.Car==0]
df2 = dataset[dataset.Car==1]
plt.scatter(df1.Age,df1['Income'],color='green')
plt.scatter(df2.Age,df2['Income'],color='red')




# In[25]:


X = dataset[['Age','Income']]
y = dataset[['Car']]


# In[ ]:





# In[ ]:





# In[34]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 5)


# In[56]:


print(X_train.count())
print(X_train)


# In[28]:


X_test


# In[14]:


y_train


# In[16]:


y_test


# In[47]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(X_train, y_train)


# In[49]:


classifier.predict([[33,149000]])


# In[48]:


classifier.score(X_test,y_test)


# In[ ]:




