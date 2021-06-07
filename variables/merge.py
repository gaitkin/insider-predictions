#!/usr/bin/env python
# coding: utf-8



#Import necessary modules
import pandas as pd

#Import variable CSV's as dataframes
df_dividends = pd.read_csv ("dividendsum.csv")
df_SP_price = pd.read_csv ("spprice.csv")
df_SP_vol = pd.read_csv ("spvol.csv")
df_interest = pd.read_csv ("spiii.csv")
df_volatility = pd.read_csv ("volatility.csv")
df_trade = pd.read_csv ("tradechange.csv")
df_amihud = pd.read_csv ("amihud.csv")
df_finali = pd.read_csv ("finaligood.csv")




#Drop empty and irrelevant columns
df_interest = df_interest.drop(columns = "Unnamed: 0")
df_SP_price = df_SP_price.drop(columns = "Unnamed: 0")
df_SP_vol = df_SP_vol.drop(columns = "Unnamed: 0")
df_trade = df_trade.drop(columns = "Unnamed: 0")
df_amihud = df_amihud.drop(columns = ["Unnamed: 0","summy","county"])
df_finali = df_finali.drop(columns = "Unnamed: 0")



#Rename columns
df_interest.rename(columns={'percentage': 'interest'}, inplace=True)
df_SP_price.rename(columns={'percentage': 'SPprice'}, inplace=True)
df_SP_vol.rename(columns={'percentage': 'SPvol'}, inplace=True)
df_volatility.rename(columns={'returns': 'volatility'}, inplace=True)
df_finali.rename(columns={'percentage': 'returns'}, inplace=True)




#Merge dataframes
julia = pd.merge(left=df_finali, right=df_interest, how="left", left_on=["stamp"], right_on=["stamp"])
julia = pd.merge(left=julia, right=df_SP_price, how="left", left_on=["stamp"], right_on=["stamp"])
julia = pd.merge(left=julia, right=df_SP_vol, how="left", left_on=["stamp"], right_on=["stamp"])
julia = pd.merge(left=julia, right=df_dividends, how="left", left_on=["id","stamp"], right_on=["id","stamp"])
julia = pd.merge(left=julia, right=df_volatility, how="left", left_on=["id","stamp"], right_on=["id","stamp"])
julia = pd.merge(left=julia, right=df_trade, how="left", left_on=["stamp"], right_on=["stamp"])
julia = pd.merge(left=julia, right=df_amihud, how="left", left_on=["id","stamp"], right_on=["id","stamp"])




#Save as CSV
julia.to_csv ("julia.csv", index = False)

