#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import matplotlib.pyplot as mtp


# In[2]:


df = pd.read_csv("C:/Users/User/OneDrive/Desktop/AI/Predict/CO2 Emissions_Canada.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


cdf = df[['Engine Size(L)','Cylinders','Fuel Consumption Comb (L/100 km)','CO2 Emissions(g/km)']]


# In[7]:


x = cdf.iloc[:, :3]


# In[8]:


y = cdf.iloc[:, -1]


# In[9]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[10]:


regressor.fit(x, y)


# In[11]:


pickle.dump(regressor, open('model.pkl','wb'))


# In[12]:


model = pickle.load(open('model.pkl','rb'))


# In[13]:


print(model.predict([[2.6, 8, 10.1]]))


# In[14]:


EngineSize = df["Engine Size(L)"]
Emission = df["CO2 Emissions(g/km)"]

mtp.scatter(EngineSize,Emission)

