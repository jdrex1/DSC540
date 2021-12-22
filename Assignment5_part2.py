#!/usr/bin/env python
# coding: utf-8

# # Import all modules needed

# In[46]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import metrics, model_selection, preprocessing
from sklearn.tree import export_text
from IPython.display import Image,display
import matplotlib.pyplot as plt
from fuzzytree import FuzzyDecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


# In[3]:


#import data via csv file
data = pd.read_csv("C:/Users/jdrex/OneDrive/Documents/DSC540/Assignment5/assignment5data.csv")
data


# In[64]:


#convert data into data frame
df=data
#convert categories into numerical values
df.temperature = pd.Categorical(df.temperature)
df['temperature_code'] = df.temperature.cat.codes

df.wind = pd.Categorical(df.wind)
df['wind_code'] = df.wind.cat.codes

df.trafficjam = pd.Categorical(df.trafficjam)
df['trafficjam_code'] = df.trafficjam.cat.codes

df.cardriving = pd.Categorical(df.cardriving)
df['cardriving_code'] = df.cardriving.cat.codes

#create df_new that drops all original columns (prior to having them converted to code)
df_new=df.drop(['days','temperature','wind','trafficjam','cardriving'],axis=1)
df_new.head()

#Create x and y values
y=df_new['cardriving_code']

x=df.drop('cardriving',axis=1)

x.shape,y.shape


# In[65]:


y.info()


# # Create Fuzzy Decision Tree

# In[61]:


clf_fuzz = FuzzyDecisionTreeClassifier().fit(x,y)
clf_sk = DecisionTreeClassifier().fit(x,y)


# In[62]:


print(f"fuzzytree: {clf_fuzz.score(x,y)}")
print(f"  sklearn: {clf_sk.score(x,y)}")


# In[63]:


plot_decision_regions(X='x', y='y', clf=clf_fuzz, legend=2)
plt.title("Fuzzy Decision Tree")
plt.xlabel("Yes")
plt.ylabel("No")

plt.show()


# In[ ]:




