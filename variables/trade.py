#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import libraries used
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

#Import trading data CSV as dataframe
df_trade = pd.read_csv ("tradedata.csv")
#Convert stamp to datetime format
df_trade["stamp"] = pd.to_datetime(df_trade["stamp"])


# In[5]:


delta_days = 30
start_date = datetime(2010, 1, 5)
now = datetime.now()
iterations = int((now - start_date).days / delta_days)

#Create empty dataframe
transactions = {
    "stamp": [],
    "trade": [],
}


#Create 30-day change in trade variable
for i in range(iterations):
    begin_transaction_date = (start_date + timedelta(days=delta_days * i)).strftime("%Y-%m-%d")
    end_transaction_date = (start_date + timedelta(days=delta_days * (i + 1))).strftime("%Y-%m-%d")
    
    after_start_date = df_trade["stamp"] >= begin_transaction_date
    before_end_date = df_trade["stamp"] <= end_transaction_date
    
    between_two_dates = after_start_date & before_end_date
    filtered_dates = df_trade.loc[between_two_dates]

    if len(list(filtered_dates.values)) == 0:
        continue

    first_elements = filtered_dates.iloc[0]
    last_elements = filtered_dates.iloc[-1]
    
    if float(first_elements.values[1]) == 0:
        continue    
        
    percentage = (
            last_elements.values[1] - first_elements.values[1]
        ) / first_elements.values[1]
 
    transactions["stamp"].append(begin_transaction_date)
    transactions["trade"].append(percentage)

df_trading = pd.DataFrame(transactions, columns = ["stamp", "trade"])


# In[6]:


df_trading.to_csv ("tradechange.csv", index = False)

