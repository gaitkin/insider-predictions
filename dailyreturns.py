#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import libraries used
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

#Import closing and insider data CSV as dataframe
df_closing = pd.read_csv ("closingdata.csv")
df_insider = pd.read_csv ("insiderdata.csv")
#Sort closing dataset
df_closing = df_closing.sort_values(by=['id', "stamp"])


# In[8]:


#Obtain list of stocks in insider dataset and use it to filter stocks in closing dataset so that both datasets contain same stocks
identifiers = [identifier for identifier in list(set(df_insider.loc[:, "id"])) if isinstance(identifier, str)]
df_closing_modified = df_closing[df_closing['id'].str.contains(f"^{'$|^'.join(identifiers)}$",regex=True)]


# In[13]:


#List of stocks in closing dataset
stocks = df_closing_modified.groupby("id").groups.keys()
#Create empty dict
transactions = {
    "stamp": [],
    "id" : [],
    "price" : [],
    "returns": [],
}

#Create empty dataframe and set stamp and index
df_daily_returns = pd.DataFrame(transactions, columns = ["stamp", "id", "price", "returns"])
df_daily_returns = df_daily_returns.set_index(["stamp"])
    
#Calculate daily returns
for stock in stocks:
    
    df_individual = df_closing_modified.groupby(["id"]).get_group(stock)
    df_individual["stamp"] = pd.to_datetime(df_individual["stamp"])
    df_individual = df_individual.set_index(["stamp"])
    daily_returns = df_individual['price'].pct_change()

    df_individual['returns'] = pd.Series(daily_returns, index=df_individual.index)

    df_daily_returns = pd.concat([df_daily_returns, df_individual], ignore_index=False)
    print(stock)


# In[14]:


#Save to CSV
df_daily_returns.to_csv ("dailyreturns.csv")

