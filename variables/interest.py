#!/usr/bin/env python
# coding: utf-8



#Import necessary modules
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

#Import interest data CSV as dataframe
df_interest = pd.read_csv ("interestdata.csv")




#Compute monthly change in short term interest rate

delta_days = 30
start_date = datetime(2010, 1, 5)
now = datetime.now()
iterations = int((now - start_date).days / delta_days)

transactions = {
    "stamp": [],
    "percentage": [],
}


yields_index = list(df_interest.columns).index("shortYield")


for i in range(iterations):
    begin_transaction_date = (start_date + timedelta(days=delta_days * i)).strftime("%Y-%m-%d")
    end_transaction_date = (start_date + timedelta(days=delta_days * (i + 1))).strftime("%Y-%m-%d")
    
    after_start_date = df_interest["stamp"] >= begin_transaction_date
    before_end_date = df_interest["stamp"] <= end_transaction_date
    
    between_two_dates = after_start_date & before_end_date
    filtered_dates = df_interest.loc[between_two_dates]

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

df_interest_change = pd.DataFrame(transactions, columns = ["stamp", "percentage"])




#Save dataframe as CSV
df_interest_change.to_csv ("sinterest.csv")

