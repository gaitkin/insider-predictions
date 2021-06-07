#!/usr/bin/env python
# coding: utf-8


#Import necessary modules
import pandas as pd
import numpy as np
import math
from scipy import stats
from datetime import datetime, timedelta


#Import insider/closing data CSV as dataframe
df_insider = pd.read_csv ("insiderdata.csv")
df_closing = pd.read_csv ("closingdata.csv")




#ARRANGE INSIDER DATASET

#Drop rows with NA
df_insider = df_insider.dropna(subset = ["id","transactionPricePerShare"])
#Add transactionValue column to insider dataset
df_insider['transactionValue'] = df_insider.apply(lambda row: row["transactionPricePerShare"] * row["transactionShares"], axis = 1)
#Set stamp column as datetime type
df_insider["stamp"] = pd.to_datetime(df_insider["stamp"])
#Set stamp column as index
df_insider = df_insider.set_index(["stamp"])




#ARRANGE PRICING DATASET

#Set stamp column as datetime type
df_closing["stamp"] = pd.to_datetime(df_closing["stamp"])


# In[14]:


#Calculate monthly sum of transactionValue for each type of transaction
delta_days = 30
start_date = datetime(2010, 1, 5)
now = datetime.now()
iterations = int((now - start_date).days / delta_days)

transactions = {
    "id": [],
    "stamp": [],
    "transactionAcquiredDisposedCode": [],
    "transactionValue": []
}

id_index = list(df_insider.columns).index("id")
transaction_index = list(df_insider.columns).index("transactionAcquiredDisposedCode")
transaction_share_index = list(df_insider.columns).index("transactionValue")

for i in range(iterations):
    begin_transaction_date = (start_date + timedelta(days=delta_days * i)).strftime("%Y-%m-%d")
    end_transaction_date = (start_date + timedelta(days=delta_days * (i + 1))).strftime("%Y-%m-%d")
    df2 = df_insider.loc[
        begin_transaction_date: end_transaction_date
    ].groupby(["id","transactionAcquiredDisposedCode"], as_index=False).sum()

    transactions["id"] += list(df2.iloc[:, id_index])
    transactions["stamp"] += [begin_transaction_date] * len(list(df2.iloc[:, id_index]))
    transactions["transactionAcquiredDisposedCode"] += list(df2.iloc[:, transaction_index])
    transactions["transactionValue"] += list(df2.iloc[:, transaction_share_index])
    
df_insider_modified = pd.DataFrame(transactions, columns = ["id", "stamp", "transactionAcquiredDisposedCode", "transactionValue"])
df_insider_modified = df_insider_modified.sort_values(by=['id', "stamp"])




#CALCULATIE MAIN REGRESSOR

#Create dispose and acquire groups
dispose = df_insider_modified.groupby("transactionAcquiredDisposedCode").get_group("D")
acquire = df_insider_modified.groupby("transactionAcquiredDisposedCode").get_group("A")
#Merge both groups to obain a separated dataset
mergy = pd.merge(left=dispose, right=acquire, how="outer", left_on=["id","stamp"], right_on=["id","stamp"], suffixes=('_D', '_A'))
#Sort values
mergy = mergy.sort_values(by=['id', "stamp"])
#Replace NA with 0
mergy['transactionValue_D'] = mergy["transactionValue_D"].fillna(0)
mergy['transactionValue_A'] = mergy["transactionValue_A"].fillna(0)
#Drop rows that create a division by 0
mergy.drop(mergy[(mergy['transactionValue_A'] + mergy["transactionValue_D"] == 0)].index , inplace=True)
#Add coefficient column
mergy['coeff'] = mergy.apply(lambda row: (row["transactionValue_A"] - row["transactionValue_D"]) / (row["transactionValue_A"] + row["transactionValue_D"]), axis = 1)




#Obtain list of stocks
identifiers = [identifier for identifier in list(set(df_insider.loc[:, "id"])) if isinstance(identifier, str)]




#Calculate monthly stock returns

from datetime import datetime, timedelta

delta_days = 30
start_date = datetime(2010, 1, 5)
now = datetime.now()
iterations = int((now - start_date).days / delta_days)

transactions = {
    "id": [],
    "stamp": [],
    "percentage": [],
}

id_index = list(df_closing.columns).index("id")

list_elements = df_closing[df_closing['id'].str.contains(f"^{'$|^'.join(identifiers)}$",regex=True)]

for i in range(iterations):
    begin_transaction_date = (start_date + timedelta(days=delta_days * i)).strftime("%Y-%m-%d")
    end_transaction_date = (start_date + timedelta(days=delta_days * (i + 1))).strftime("%Y-%m-%d")
    
    after_start_date = list_elements["stamp"] >= begin_transaction_date
    before_end_date = list_elements["stamp"] <= end_transaction_date
    
    between_two_dates = after_start_date & before_end_date
    filtered_dates = list_elements.loc[between_two_dates]
    filtered_dates = filtered_dates.set_index("id")

    if len(list(filtered_dates.values)) == 0:
        continue

    first_elements = filtered_dates.groupby("id").first()
    last_elements = filtered_dates.groupby("id").last()
    for index_filter in range(len(list(filtered_dates.groupby("id").first().values))):
        percentage = (
            last_elements.values[index_filter][1] - first_elements.values[index_filter][1]
        ) / first_elements.values[index_filter][1] 

        transactions["id"].append(first_elements.index[index_filter])
        transactions["stamp"].append(begin_transaction_date)
        transactions["percentage"].append(percentage)

df_price_change = pd.DataFrame(transactions, columns = ["id", "stamp", "percentage"])
df_price_change = df_price_change.set_index('id')
df_price_change = df_price_change.sort_values(by=['id', "stamp"])




#Select relevant columns
selective = mergy[["id","stamp","coeff"]]

#Merge dataframes
finali = pd.merge(left=selective, right=df_price_change, how="left", left_on=["id","stamp"], right_on=["id","stamp"])




#Save as CSV
finali.to_csv ("finaligood.csv", index = False)

