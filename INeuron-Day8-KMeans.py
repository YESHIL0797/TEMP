#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os as o
os.getcwd()
o.chdir('D:/Project/Ineuron')


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
df = pd.read_csv("Book.csv")
df.head(50)


# In[15]:


plt.scatter(df.rollno,df['marks'])
plt.xlabel('rollno')
plt.ylabel('marks')


# In[16]:


plt.scatter(df.rollno,df.marks)


# In[21]:


Km = KMeans(n_clusters=3)
predicted = Km.fit_predict(df[['rollno','marks']])
predicted


# In[23]:


df['cluster']=predicted
df.head()


# In[24]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.rollno,df1['marks'],color='green')
plt.scatter(df2.rollno,df2['marks'],color='red')
plt.scatter(df3.rollno,df3['marks'],color='black')
plt.xlabel('rollno')
plt.ylabel('marks')


# In[25]:


scale = MinMaxScaler()

scale.fit(df[['marks']])
df['marks'] = scale.transform(df[['marks']])

scale.fit(df[['rollno']])
df['rollno'] = scale.transform(df[['rollno']])
df


# In[30]:


km = KMeans(n_clusters=3)
km.fit(df[['rollno','marks']])
predicted = km.fit_predict(df[['rollno','marks']])
predicted


# In[27]:


df = df.drop(['cluster'], axis='columns')

df['cluster']=predicted
df.head()


# In[28]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.rollno,df1['marks'],color='green')
plt.scatter(df2.rollno,df2['marks'],color='red')
plt.scatter(df3.rollno,df3['marks'],color='black')
plt.xlabel('rollno')
plt.ylabel('marks')


# In[29]:


km.cluster_centers_


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




