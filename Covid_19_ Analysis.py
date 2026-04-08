#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Load Data
df = pd.read_csv("covid_19_cleaned.csv")


# In[6]:


# Clean
df.columns = df.columns.str.lower()


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


# Create Metrics
df["cases_per_100k"] = (df["cases"]/df["population"])*100000
df["death_rate"] = (df["deaths"]/df["cases"])*100
df["recovery_rate"] = (df["recovered"]/df["cases"])*100


# In[10]:


# Correlation
corr = df.corr()


# In[11]:


plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.title("Correlation Matrix")
plt.show()


# In[12]:


# Top Death Rate
top_death = df.sort_values("death_rate", ascending=False).head(10)

plt.figure()
plt.barh(top_death["country"], top_death["death_rate"])
plt.title("Top Death Rate")
plt.show()


# In[13]:


# Recovery Rate
top_recovery = df.sort_values("recovery_rate", ascending=False).head(10)

plt.figure()
plt.barh(top_recovery["country"], top_recovery["recovery_rate"])
plt.title("Recovery Rate")
plt.show()


# In[14]:


# Scatter Plots
plt.figure()
plt.scatter(df["population"], df["cases"])
plt.title("Population vs Cases")
plt.show()


# In[15]:


plt.figure()
plt.scatter(df["tests"], df["cases"])
plt.title("Tests vs Cases")
plt.show()


# # Highest Death Rate Countries
# 
# Top High-Risk Countries:
# 
# MS-Zaandam
# Yemen
# Western Sahara
# Sudan
# Syria
# Somalia
# Peru
# Egypt
# Mexico
# Bosnia & Herzegovina
# 
# Interpretation:
# 
# These regions likely have:
# 
# Limited healthcare
# Low testing
# Underreporting

# # Highest Recovery Rate Countries
# 
# Top Recovery Countries:
# 
# Palau
# Cyprus
# Taiwan
# Qatar
# Marshall Islands
# South Korea
# Singapore
# Vatican City
# Falkland Islands
# 
# Insight:
# 
# These countries typically have:
# 
# Strong healthcare
# Strong testing
# Early containment
