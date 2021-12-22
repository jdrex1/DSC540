#!/usr/bin/env python
# coding: utf-8

# # Import tools

# In[23]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import metrics, model_selection, preprocessing
from sklearn.tree import export_text
from IPython.display import Image,display
import matplotlib.pyplot as plt


# # Import data and format data correctly

# In[3]:


#import data via csv file
data = pd.read_csv("C:/Users/jdrex/OneDrive/Documents/DSC540/Assignment5/assignment5data.csv")
data


# In[4]:


#convert data into data frame
df=data


# In[5]:


#convert categories into numerical values
df.temperature = pd.Categorical(df.temperature)
df['temperature_code'] = df.temperature.cat.codes

df.wind = pd.Categorical(df.wind)
df['wind_code'] = df.wind.cat.codes

df.trafficjam = pd.Categorical(df.trafficjam)
df['trafficjam_code'] = df.trafficjam.cat.codes

df.cardriving = pd.Categorical(df.cardriving)
df['cardriving_code'] = df.cardriving.cat.codes

df.info()


# In[6]:


df.head()


# In[7]:


df_new=df.drop(['days','temperature','wind','trafficjam','cardriving'],axis=1)
df_new.head()


# In[8]:


#Create x and y values
y=df['cardriving']

x=df_new.drop('cardriving_code',axis=1)

x.shape,y.shape


# # Create & Fit Model

# In[9]:


model_tree = tree.DecisionTreeClassifier(criterion='entropy',max_depth=100)
model_tree = model_tree.fit(x, y)
plt.figure(figsize=(12,12))
tree.plot_tree(model_tree,feature_names=['temperature','wind','trafficjam'],class_names=['yes','no'],fontsize=12)
...


# In[10]:


r = export_text(model_tree)
print(r)


# In[11]:


y_pred=model_tree.predict(x)
print(y_pred,y)


# # Part 2- Fuzzy Decisions

# In[29]:


from fuzzytree import FuzzyDecisionTreeClassifier


# In[ ]:





# In[ ]:




