#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn import svm
import matplotlib.pyplot as plt


# (a)	Generate 50 data points from this function in the range [– 3, 3]. 

# In[53]:


data=np.linspace(-3,3,50)  #Using reference numpy.linspace & numpy.sinc from numpy.org
sincfunc=np.sinc(data)


# In[54]:


plt.plot(data,sincfunc,marker="1")
plt.title("sinc function")


# (b)	Add Gaussian noise to the data.

# In[55]:


gausnoise=np.random.normal(loc=0,scale=.25,size=50)
sincfunc_gausnoise=(sincfunc+gausnoise)


# In[56]:


plt.plot(data,sincfunc_gausnoise,marker="1")
plt.title("sinc function with noise added")


# (c)	Train an SVM (c) regressor with the data generated in (a). Define (and explain) suitable parameters required for training the regressor. 

# In[37]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# In[57]:


#for below code, used https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html example, but was unable to get it to work as needed

#n_samples, n_features = 50, 5
#ng = np.random.RandomState(0)
#y = rng.randn(n_samples)
#X = rng.randn(n_samples, n_features)
#regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
#regr.fit(X, y)
#Pipeline(steps=[('standardscaler', StandardScaler()),
                #('svr', SVR(epsilon=0.2))])


# (d)	Describe the functionality of the regressor.
# (e)	Discuss the potential use of the regressor and quantify its accuracy.

# In general, the functionality of a regressor can be described as "attempts to describe the strength and character of the relationship between one independent variable and a series of other variables"(Beers). 

# References: 
# 
# Numpy.Linspace. Numpy. API Reference. Retrieved from: https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
# Numpy.Sinc. Numpy. API Reference. Retrieved from: https://numpy.org/doc/stable/reference/generated/numpy.sinc.html
# 
# Beers, B. October 30, 2021. Regression Definiton. Investopedia. Retrieved from: https://www.investopedia.com/terms/r/regression.asp
# 
# 
