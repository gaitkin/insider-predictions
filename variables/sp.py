#!/usr/bin/env python
# coding: utf-8



#Import necessary modules
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf


#EXTRA: download S&P data
sp = yf.download('^GSPC', start='2010-01-04', end='2020-01-01')

sp.to_csv ("SPdata.csv")




#Import SP data CSV as dataframe
df_sp = pd.read_csv ("SPdata.csv")
#Divide into price and volume dataframes
df_sp_price = df_sp[["stamp","Close"]]
df_sp_volume = df_sp[["stamp","Volume"]]




#Calculate monthly change in S&P price

delta_days = 30
start_date = datetime(2010, 1, 5)
now = datetime.now()
iterations = int((now - start_date).days / delta_days)

transactions = {
    "stamp": [],
    "percentage": [],
}


for i in range(iterations):
    begin_transaction_date = (start_date + timedelta(days=delta_days * i)).strftime("%Y-%m-%d")
    end_transaction_date = (start_date + timedelta(days=delta_days * (i + 1))).strftime("%Y-%m-%d")
    
    after_start_date = df_sp_price["stamp"] >= begin_transaction_date
    before_end_date = df_sp_price["stamp"] <= end_transaction_date
    
    between_two_dates = after_start_date & before_end_date
    filtered_dates = df_sp_price.loc[between_two_dates]

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
    transactions["percentage"].append(percentage)

df_sp_price_change = pd.DataFrame(transactions, columns = ["stamp", "percentage"])




#Calculate monthly change in S&P volume

delta_days = 30
start_date = datetime(2010, 1, 5)
now = datetime.now()
iterations = int((now - start_date).days / delta_days)

transactions = {
    "stamp": [],
    "percentage": [],
}


for i in range(iterations):
    begin_transaction_date = (start_date + timedelta(days=delta_days * i)).strftime("%Y-%m-%d")
    end_transaction_date = (start_date + timedelta(days=delta_days * (i + 1))).strftime("%Y-%m-%d")
    
    after_start_date = df_sp_volume["stamp"] >= begin_transaction_date
    before_end_date = df_sp_volume["stamp"] <= end_transaction_date
    
    between_two_dates = after_start_date & before_end_date
    filtered_dates = df_sp_volume.loc[between_two_dates]

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
    transactions["percentage"].append(percentage)

df_sp_volume_change = pd.DataFrame(transactions, columns = ["stamp", "percentage"])




#Save as CSV
df_sp_price_change.to_csv ("C:/Users/PortatilUPF/Downloads/spvol.csv", index = False)
df_sp_volume_change.to_csv ("C:/Users/PortatilUPF/Downloads/spvol.csv", index = False)

