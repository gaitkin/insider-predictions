#!/usr/bin/env python
# coding: utf-8



#Import libraries used
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from yahoofinancials import YahooFinancials
#Import dividends data CSV as dataframe
df_dividends = pd.read_csv ("dividenddata.csv")
#Convert stamp to datetime
df_dividends["stamp"] = pd.to_datetime(df_dividends["stamp"])
#Set stamp as index
df_dividends = df_dividends.set_index(["stamp"])




#EXTRA: Dividend data download
#Create list of stocks
stock_list = list(df.groupby(["id"]).groups.keys())
#Create empty dict
transactions = {
    "formatted_date": [],
    "amount": [],
    "ticker": [],
}
#Create empty dataframe
df_dividends_download = pd.DataFrame(transactions, columns = ["formatted_date"])
    
    
for stock in stock_list:
    yahoo_financials = YahooFinancials(stock)
    
    data = yahoo_financials.get_daily_dividend_data(start_date='2010-01-04', 
                                                  end_date='2019-12-31')
    if data[stock] == None:
        continue
    else:
        df_a = pd.DataFrame(data[stock])
        df_a = df_a.drop('date', axis=1)
        columni = stock
        df_a["id"] = columni
        df_dividends_download = pd.concat([df_dividends_download, df_a], ignore_index=True)
        print(stock)




#Arrange downloaded dataframe
#Drop rows with NA
df_dividends_download = df_dividends_download.dropna(subset = ["id","dividends"])
#Remove stamp time
df_dividends_download = df_dividends_download.rename(columns = {"formatted_date" : "stamp"}, inplace = False)
#Reset index
df_dividends_download = df_dividends_download.reset_index(drop=True)
#Convert stamp to datetime
df_dividends_download["stamp"] = pd.to_datetime(df_dividends_download["stamp"])
#Set stamp column as index
df_dividends_download = df_dividends_download.set_index(["stamp"])




#Monthly aggregation of dividends
delta_days = 30
start_date = datetime(2010, 1, 5)
now = datetime.now()
iterations = int((now - start_date).days / delta_days)

#Create empty dict
transactions = {
    "id": [],
    "stamp": [],
    "dividends": [],
}

id_index = list(df_dividends.columns).index("id")
transaction_index = list(df_dividends.columns).index("dividends")

#Aggregate dividends
for i in range(iterations):
    begin_transaction_date = (start_date + timedelta(days=delta_days * i)).strftime("%Y-%m-%d")
    end_transaction_date = (start_date + timedelta(days=delta_days * (i + 1))).strftime("%Y-%m-%d")
    df2 = df_dividends.loc[
        begin_transaction_date: end_transaction_date
    ].groupby(["id"], as_index=False).sum()

    transactions["id"] += list(df2.iloc[:, id_index])
    transactions["stamp"] += [begin_transaction_date] * len(list(df2.iloc[:, id_index]))
    transactions["dividends"] += list(df2.iloc[:, transaction_index])
    
df_dividends_sum = pd.DataFrame(transactions, columns = ["id", "stamp", "dividends"])
df_dividends_sun = df_dividends_sum.set_index('id')
df_dividends_sum = df_dividends_sum.sort_values(by=['id', "stamp"])




#Save as CSV
df_dividends_sum.to_csv("dividendsum.csv")

