#!/usr/bin/env python
# coding: utf-8


#Import libraries used
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

#Import daily returns data CSV as dataframe
df_daily_returns = pd.read_csv ("dailyreturns.csv")
#Tranform stamp to datetime
df_daily_returns["stamp"] = pd.to_datetime(df_daily_returns["stamp"])
#Set stamp as index
df_daily_returns = df_daily_returns.set_index(["stamp"])




#Calculate volatility
delta_days = 30
start_date = datetime(2010, 1, 5)
now = datetime.now()
iterations = int((now - start_date).days / delta_days)

#Create empty dataframe
transactions = {
    "id" : [],
    "stamp": [],
    "returns": [],
}

id_index = list(df_daily_returns.columns).index("id")
returns_index = list(df_daily_returns.columns).index("returns")

for i in range(iterations):
    begin_transaction_date = (start_date + timedelta(days=delta_days * i)).strftime("%Y-%m-%d")
    end_transaction_date = (start_date + timedelta(days=delta_days * (i + 1))).strftime("%Y-%m-%d")
    df2 = df_daily_returns.loc[
        begin_transaction_date: end_transaction_date
    ].groupby(["id"], as_index=False).std()

    transactions["id"] += list(df2.iloc[:, id_index])
    transactions["stamp"] += [begin_transaction_date] * len(list(df2.iloc[:, id_index]))
    transactions["returns"] += list(df2.iloc[:, returns_index])
    
df_volatility = pd.DataFrame(transactions, columns = ["id", "stamp", "returns"])
df_volatility = df_volatility.sort_values(by=['id', "stamp"])




#Save as CSV
df_volatility.to_csv ("volatility.csv", index = False)

