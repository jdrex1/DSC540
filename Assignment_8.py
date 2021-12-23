#!/usr/bin/env python
# coding: utf-8

# # Import tools

# In[37]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score


# # Import data 

# In[2]:


activities_performed = pd.read_csv("C:/Users/jdrex/OneDrive/Documents/DSC540/Assignment8/activities_performed.csv")
subject_info = pd.read_csv("C:/Users/jdrex/OneDrive/Documents/DSC540/Assignment8/Subject_information.csv")
df_performed=pd.DataFrame(activities_performed)
df_info=pd.DataFrame(subject_info)


# In[3]:


df_performed


# In[4]:


df_info.head()


# # Data Preprocessing

# In[5]:


#slicing data into independent and dependent variables
y=df_performed['activity_num']
x=df_performed.drop('activity_num',axis=1)
x=x.drop('Activity',axis=1)
x=x.drop('nr_of_subjects',axis=1)


# In[6]:


x


# In[7]:


y


# # Classification

# Using k-nearest neighbors classifier- https://www.analyticsvidhya.com/blog/2021/01/a-quick-introduction-to-k-nearest-neighbor-knn-classification-using-python/

# In[12]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
clf=classifier.fit(x, y)


# In[9]:


z=classifier.predict(x)

pd.crosstab(z,y)


# # Decision Fusion Technique

# In[13]:


bagging = BaggingClassifier(KNeighborsClassifier(),
                        max_samples=0.5, max_features=0.5)


# In[40]:


x, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
model=BaggingClassifier()
model.fit(x, y)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')


# In[39]:


# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# In[ ]:




