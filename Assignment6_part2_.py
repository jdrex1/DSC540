#!/usr/bin/env python
# coding: utf-8

# # Import tools

# In[33]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import metrics, model_selection, preprocessing
from sklearn.tree import export_text
from IPython.display import Image,display
import matplotlib.pyplot as plt


# # Problem 1

# In[14]:


x1=-10
x2=-10
y1=10
y2=10
print ("[x1,y1]", [x1,y1])
print ("[x2,y2]", [x2,y2])


# In[26]:


y=((np.sinc(x1))*(np.sinc(x2)))/(x1*x2)
y


# # Problem 2

# In[ ]:


import math
x1=1
x2=1
x3=1
p1=1
p2=1
y=(1+(x1)**p1+(x2)^-1+(x3)^p2)^2
y


# In[28]:


train_data= [1,6],[1,6],[1,6]
test_data= [1.5,5.5],[1.5,5.5],[1.5,5.5]


# Below is the code from reference 
# Gafa, C. Fuzzy Inference System Implementation in Python. Towards Data Science. Retrieved from: https://towardsdatascience.com/fuzzy-inference-system-implementation-in-python-8af88d1f0a6e
# 
# 

# In[62]:


#creating three (a,b,c triangular domains)
def create_triangular(cls, name, domain_min, domain_max, res, a, b, c):
  t1fs = cls(name, domain_min, domain_max, res)

  a = t1fs._adjust_domain_val(a)
  b = t1fs._adjust_domain_val(b)
  c = t1fs._adjust_domain_val(c)

  t1fs._dom = np.round(np.maximum(np.minimum((t1fs._domain-a)/(b-a), (c-t1fs._domain)/(c-b)), 0), t1fs._precision)


# In[63]:


def _adjust_domain_val(self, x_val):
  return self._domain[np.abs(self._domain-x_val).argmin()]


# In[65]:


def create_triangular(cls, name, domain_min, domain_max, res, a, b, c):
  t1fs = cls(name, domain_min, domain_max, res)

  a = t1fs._adjust_domain_val(a)
  b = t1fs._adjust_domain_val(b)
  c = t1fs._adjust_domain_val(c)

  t1fs._dom = np.round(np.maximum(np.minimum((t1fs._domain-a)/(b-a), (c-t1fs._domain)/(c-b)), 0), t1fs._precision)


# In[73]:


def union(self, f_set):

		result = FuzzySet(f'({self._name}) union ({f_set._name})', self._domain_min, self._domain_max, self._res)
		result._dom = np.maximum(self._dom, f_set._dom)

		return result


# In[76]:


def cog_defuzzify(self):

  num = np.sum(np.multiply(self._dom, self._domain))
  den = np.sum(self._dom)

  return num/den


# In[77]:


def fuzzify(self, val):

	# get dom for each set and store it - 
	# it will be required for each rule
	for set_name, f_set in self._sets.items():
		f_set.last_dom_value = f_set[val]


# In[79]:


class FuzzyOutputVariable(FuzzyVariable):

    def __init__(self, name, min_val, max_val, res):
        super().__init__(name, min_val, max_val, res)
        self._output_distribution = FuzzySet(name, min_val, max_val, res)

    def add_rule_contribution(self, rule_consequence):
        self._output_distribution = self._output_distribution.union(rule_consequence)

    def get_crisp_output(self):
        return self._output_distribution.cog_defuzzify()


# In[80]:


# execution methods for a FuzzyClause
# that comtains a FuzzyVariable; _variable
# and a FuzzySet; _set
  
def evaluate_antecedent(self):
	return self._set.last_dom_value

def evaluate_consequent(self, activation):
	self._variable.add_rule_contribution(self._set.min_scalar(activation))


# In[81]:


def evaluate(self):
	# rule activation initialize to 1 as min operator will be performed
	rule_activation = 1
	# execute all antecedent clauses, keeping the minimum of the returned doms to determine the activation
	for ante_clause in self._antecedent:
		rule_activation = min(ante_clause.evaluate_antecedent(), rule_activation)

	# execute consequent clauses, each output variable will update its output_distribution set
	for consequent_clause in self._consequent:
		consequent_clause.evaluate_consequent(rule_activation)


# In[ ]:




