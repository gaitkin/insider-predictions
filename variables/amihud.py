#!/usr/bin/env python
# coding: utf-8



#Import libraries used
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

#Import closing/daily returns data CSV as dataframe
df_closing = pd.read_csv ("closingdata.csv")
df_daily_returns = pd.read_csv ("dailyreturns.csv")

#Select relevant columns used in closing dataset
df_closing = df_closing[["id","stamp","value"]] 




#Merge both dataframes
df_merged = pd.merge(left=df_daily_returns, right=df_closing, how="left", left_on=["id","stamp"], right_on=["id","stamp"])

#Drop NA & 0 values
df_merged = df_merged.dropna(subset = ["returns"])
df_merged.drop(df_merged[(df_merged["value"] == 0)].index , inplace=True)

#Create Amihud column with first step of calculations
df_merged['amihud'] = df_merged.apply(lambda row: abs(row["returns"]) / row["value"], axis = 1)

#Transform stamp to datetime
df_merged["stamp"] = pd.to_datetime(df_merged["stamp"])
#Set stamp column as index
df_merged = df_merged.set_index(["stamp"])




delta_days = 30
start_date = datetime(2010, 1, 5)
now = datetime.now()
iterations = int((now - start_date).days / delta_days)

#Create empty dataframe
transactions = {
    "id": [],
    "stamp": [],
    "summy": [],
    "county": [],
}

id_index = list(df_merged.columns).index("id")
amihud_index = list(df_merged.columns).index("amihud")

#Next stetp of Amihud calculations
for i in range(iterations):
    begin_transaction_date = (start_date + timedelta(days=delta_days * i)).strftime("%Y-%m-%d")
    end_transaction_date = (start_date + timedelta(days=delta_days * (i + 1))).strftime("%Y-%m-%d")
    df2 = df_merged.loc[
        begin_transaction_date: end_transaction_date
    ].groupby(["id"], as_index=False).sum()
    df3 = df_merged.loc[
        begin_transaction_date: end_transaction_date
    ].groupby(["id"], as_index=False).count()

    transactions["id"] += list(df2.iloc[:, id_index])
    transactions["stamp"] += [begin_transaction_date] * len(list(df2.iloc[:, id_index]))
    transactions["summy"] += list(df2.iloc[:, amihud_index])
    transactions["county"] += list(df3.iloc[:, amihud_index])
    
df_amihud = pd.DataFrame(transactions, columns = ["id", "stamp", "summy", "county"])
df_amihud = df_amihud.sort_values(by=['id', "stamp"])




#Final step of Amihud calculations
df_amihud['amihud'] = df_amihud.apply(lambda row: (row["summy"]) / row["county"], axis = 1)




#Save as CSV
df_amihud.to_csv ("amihud.csv")

