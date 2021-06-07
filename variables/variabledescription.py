#!/usr/bin/env python
# coding: utf-8

# In[25]:


#Import relevant modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

julia = pd.read_csv ("julia.csv")
julia = julia.drop(columns = "Unnamed: 0")
julia['dividends'] = julia['dividends'].fillna(0)


# In[26]:


#Describe categorical values
julia[["id","stamp"]].describe()


# In[27]:


#Descibe numerical values
julia.describe()


# In[28]:


#Create correlation matrix
plt.figure(figsize=(15,10))
sns.heatmap(julia.corr(), cmap="YlGnBu", annot=True)
plt.savefig('corr.png')
plt.show()


# In[ ]:


#Create correlation scatter matrix
scatter_matrix(julia, figsize = (15,10))
plt.savefig("scat.png")
plt.show()


# In[ ]:


#Create lagged values
julia['laginterest'] = julia.groupby('id')['interest'].shift(-1)
julia['lagvolatility'] = julia.groupby('id')['volatility'].shift(-1)
julia['lagcoeff'] = julia.groupby('id')['coeff'].shift(-1)
julia['lagSPprice'] = julia.groupby('id')['SPprice'].shift(-1)
julia['lagdividends'] = julia.groupby('id')['dividends'].shift(-1)
julia['lagSPvol'] = julia.groupby('id')['SPvol'].shift(-1)
julia['lagtrade'] = julia.groupby('id')['trade'].shift(-1)
julia['lagamihud'] = julia.groupby('id')['amihud'].shift(-1)
julia['lagreturns'] = julia.groupby('id')["returns"].shift(-1)


# In[ ]:


#Create correlation matrix with lags
plt.figure(figsize=(15,10))
sns.heatmap(julia[["returns","lagcoeff","laginterest","lagSPprice","lagSPvol","lagdividends","lagvolatility","lagtrade","lagamihud"]].corr(), cmap="YlGnBu", annot=True)
plt.savefig('corrlag.png')
plt.show()


# In[21]:


#Create correlation scatter matrix with lags
scatter_matrix(julia[["returns","lagcoeff","laginterest","lagSPprice","lagSPvol","lagdividends","lagvolatility","lagtrade","lagamihud"]], figsize = (15,10))
plt.savefig("scatlag.png")
plt.show()

