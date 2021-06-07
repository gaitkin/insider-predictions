#!/usr/bin/env python
# coding: utf-8

# In[19]:


#Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
from linearmodels import PooledOLS
import statsmodels.api as sm
import matplotlib.pyplot as plt
from linearmodels import PanelOLS
from linearmodels import RandomEffects
from linearmodels.panel import compare
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pylab import *


# In[2]:


#Dataset
julia = pd.read_csv ("C:/Users/PC/Downloads/julia.csv")
julia = julia.drop(columns = "Unnamed: 0")
julia['dividends'] = julia['dividends'].fillna(0)
julia = julia.sort_values(by=['id', 'stamp'])
julia["stamp"] = pd.to_datetime(julia["stamp"])

#Add lags
julia['laginterest'] = julia.groupby('id')['interest'].shift(-1)
julia['lagvolatility'] = julia.groupby('id')['volatility'].shift(-1)
julia['lagcoeff'] = julia.groupby('id')['coeff'].shift(-1)
julia['lagSPprice'] = julia.groupby('id')['SPprice'].shift(-1)
julia['lagdividends'] = julia.groupby('id')['dividends'].shift(-1)
julia['lagSPvol'] = julia.groupby('id')['SPvol'].shift(-1)
julia['lagtrade'] = julia.groupby('id')['trade'].shift(-1)
julia['lagamihud'] = julia.groupby('id')['amihud'].shift(-1)
julia["lagreturns"] = julia.groupby('id')['returns'].shift(-1)
julia["lagreturns2"] = julia.groupby('id')['returns'].shift(-2)
julia["lagreturns3"] = julia.groupby('id')['returns'].shift(-3)
julia['lagcoeff2'] = julia.groupby('id')['coeff'].shift(-2)
julia['lagcoeff3'] = julia.groupby('id')['coeff'].shift(-3)
julia['laginterest2'] = julia.groupby('id')['interest'].shift(-2)
julia['laginterest3'] = julia.groupby('id')['interest'].shift(-3)
julia['lagvolatility2'] = julia.groupby('id')['volatility'].shift(-2)
julia['lagvolatility3'] = julia.groupby('id')['volatility'].shift(-3)
julia['lagSPprice2'] = julia.groupby('id')['SPprice'].shift(-2)
julia['lagSPprice3'] = julia.groupby('id')['SPprice'].shift(-3)
julia['lagdividends2'] = julia.groupby('id')['dividends'].shift(-2)
julia['lagdividends3'] = julia.groupby('id')['dividends'].shift(-3)
julia['lagSPvol2'] = julia.groupby('id')['SPvol'].shift(-2)
julia['lagSPvol3'] = julia.groupby('id')['SPvol'].shift(-3)
julia['lagtrade2'] = julia.groupby('id')['trade'].shift(-2)
julia['lagtrade3'] = julia.groupby('id')['trade'].shift(-3)
julia['lagamihud2'] = julia.groupby('id')['amihud'].shift(-2)
julia['lagamihud3'] = julia.groupby('id')['amihud'].shift(-3)
julia['laginterest4'] = julia.groupby('id')['interest'].shift(-4)
julia['lagvolatility4'] = julia.groupby('id')['volatility'].shift(-4)
julia['lagcoeff4'] = julia.groupby('id')['coeff'].shift(-4)
julia['lagtrade4'] = julia.groupby('id')['trade'].shift(-4)
julia['lagamihud4'] = julia.groupby('id')['amihud'].shift(-4)
julia["lagreturns4"] = julia.groupby('id')['returns'].shift(-4)
julia['laginterest5'] = julia.groupby('id')['interest'].shift(-5)
julia['lagvolatility5'] = julia.groupby('id')['volatility'].shift(-5)
julia['lagcoeff5'] = julia.groupby('id')['coeff'].shift(-5)
julia['lagtrade5'] = julia.groupby('id')['trade'].shift(-5)
julia['lagamihud5'] = julia.groupby('id')['amihud'].shift(-5)
julia["lagreturns5"] = julia.groupby('id')['returns'].shift(-5)
julia['laginterest6'] = julia.groupby('id')['interest'].shift(-6)
julia['lagvolatility6'] = julia.groupby('id')['volatility'].shift(-6)
julia['lagcoeff6'] = julia.groupby('id')['coeff'].shift(-6)
julia['lagtrade6'] = julia.groupby('id')['trade'].shift(-6)
julia['lagamihud6'] = julia.groupby('id')['amihud'].shift(-6)
julia["lagreturns6"] = julia.groupby('id')['returns'].shift(-6)
julia['laginterest7'] = julia.groupby('id')['interest'].shift(-7)
julia["lagvolatility7"] = julia.groupby('id')['volatility'].shift(-7)
julia['lagcoeff7'] = julia.groupby('id')['coeff'].shift(-7)
julia['lagtrade7'] = julia.groupby('id')['trade'].shift(-7)
julia['lagamihud7'] = julia.groupby('id')['amihud'].shift(-7)
julia["lagreturns7"] = julia.groupby('id')['returns'].shift(-7)
julia["lagvolatility8"] = julia.groupby('id')['volatility'].shift(-8)
julia['lagcoeff8'] = julia.groupby('id')['coeff'].shift(-8)
julia['lagtrade8'] = julia.groupby('id')['trade'].shift(-8)
julia['lagamihud8'] = julia.groupby('id')['amihud'].shift(-8)
julia["lagreturns8"] = julia.groupby('id')['returns'].shift(-8)
julia["lagvolatility9"] = julia.groupby('id')['volatility'].shift(-8)
julia['lagcoeff9'] = julia.groupby('id')['coeff'].shift(-9)
julia['lagtrade9'] = julia.groupby('id')['trade'].shift(-9)
julia['lagamihud9'] = julia.groupby('id')['amihud'].shift(-9)
julia["lagreturns9"] = julia.groupby('id')['returns'].shift(-9)
julia["lagvolatility10"] = julia.groupby('id')['volatility'].shift(-9)
julia['lagcoeff10'] = julia.groupby('id')['coeff'].shift(-10)
julia['lagtrade10'] = julia.groupby('id')['trade'].shift(-10)
julia['lagamihud10'] = julia.groupby('id')['amihud'].shift(-10)
julia["lagreturns10"] = julia.groupby('id')['returns'].shift(-10)
julia["lagvolatility11"] = julia.groupby('id')['volatility'].shift(-10)
julia['lagcoeff11'] = julia.groupby('id')['coeff'].shift(-11)
julia['lagtrade11'] = julia.groupby('id')['trade'].shift(-11)
julia['lagamihud11'] = julia.groupby('id')['amihud'].shift(-11)
julia["lagreturns11"] = julia.groupby('id')['returns'].shift(-11)
julia["lagvolatility12"] = julia.groupby('id')['volatility'].shift(-12)
julia['lagcoeff12'] = julia.groupby('id')['coeff'].shift(-12)
julia['lagtrade12'] = julia.groupby('id')['trade'].shift(-12)
julia['lagamihud12'] = julia.groupby('id')['amihud'].shift(-12)
julia["lagreturns12"] = julia.groupby('id')['returns'].shift(-12)
julia['lagSPprice4'] = julia.groupby('id')['SPprice'].shift(-4)
julia['lagdividends4'] = julia.groupby('id')['dividends'].shift(-4)
julia['lagSPvol4'] = julia.groupby('id')['SPvol'].shift(-4)
julia['lagSPprice5'] = julia.groupby('id')['SPprice'].shift(-5)
julia['lagdividends5'] = julia.groupby('id')['dividends'].shift(-5)
julia['lagSPvol5'] = julia.groupby('id')['SPvol'].shift(-5)
julia['lagSPprice6'] = julia.groupby('id')['SPprice'].shift(-6)
julia['lagdividends6'] = julia.groupby('id')['dividends'].shift(-6)
julia['lagSPvol6'] = julia.groupby('id')['SPvol'].shift(-6)
julia['lagSPprice7'] = julia.groupby('id')['SPprice'].shift(-7)
julia['lagdividends7'] = julia.groupby('id')['dividends'].shift(-7)
julia['lagSPvol7'] = julia.groupby('id')['SPvol'].shift(-7)
julia['lagSPprice8'] = julia.groupby('id')['SPprice'].shift(-8)
julia['lagdividends8'] = julia.groupby('id')['dividends'].shift(-8)
julia['lagSPvol8'] = julia.groupby('id')['SPvol'].shift(-8)
julia['lagSPprice9'] = julia.groupby('id')['SPprice'].shift(-9)
julia['lagdividends9'] = julia.groupby('id')['dividends'].shift(-9)
julia['lagSPvol9'] = julia.groupby('id')['SPvol'].shift(-9)
julia['lagSPprice10'] = julia.groupby('id')['SPprice'].shift(-10)
julia['lagdividends10'] = julia.groupby('id')['dividends'].shift(-10)
julia['lagSPvol10'] = julia.groupby('id')['SPvol'].shift(-10)
julia['lagSPprice11'] = julia.groupby('id')['SPprice'].shift(-11)
julia['lagdividends11'] = julia.groupby('id')['dividends'].shift(-11)
julia['lagSPvol11'] = julia.groupby('id')['SPvol'].shift(-11)
julia['lagSPprice12'] = julia.groupby('id')['SPprice'].shift(-12)
julia['lagdividends12'] = julia.groupby('id')['dividends'].shift(-12)
julia['lagSPvol12'] = julia.groupby('id')['SPvol'].shift(-12)
julia['laginterest8'] = julia.groupby('id')['interest'].shift(-8)
julia['laginterest9'] = julia.groupby('id')['interest'].shift(-9)
julia['laginterest10'] = julia.groupby('id')['interest'].shift(-10)
julia['laginterest11'] = julia.groupby('id')['interest'].shift(-11)
julia['laginterest12'] = julia.groupby('id')['interest'].shift(-12)


# In[15]:


#Define function to cross-validate models, where params are variables to be tested

def modeling(params):

    #Train/Test split (80/20) into 5 groups

    #Find 80th percentile id and split split data accordingly
    
    stocks = julia.groupby("id").groups.keys()
    the_one = list(stocks)[int(len(stocks)*0.8)]
    splity = (julia.index[julia['id'] == the_one])
    training_data = julia.loc[0:splity[0]]
    test_data = julia.loc[splity[0]:]

    #List of dates
    stamp_list = sorted(list(set(julia.stamp)))
    #Splitting values
    values = (0.5, 0.625, 0.75, 0.875)

    def split(value):
        date1 = stamp_list[(int(len(stamp_list)*value))-1].strftime('%Y-%m-%d')
        date2 = (stamp_list[(int(len(stamp_list)*value*0.8))-1]).strftime('%Y-%m-%d')


        training_data1 = training_data[training_data["stamp"] < pd.to_datetime(date2)].copy()
        training_data1 = training_data1.set_index(["id", "stamp"])


        test_data1 = test_data[(test_data["stamp"] >= pd.to_datetime(date2)) & (test_data["stamp"] < pd.to_datetime(date1))].copy()
        test_data1 = test_data1.set_index(["id", "stamp"])

        X_train =training_data1.drop(columns=['returns'])
        X_test =test_data1.drop(columns=['returns'])
        Y_train=training_data1.drop(columns=["coeff","interest","SPprice","SPvol","dividends","volatility","trade","amihud",'laginterest','lagvolatility','lagcoeff','lagSPprice','lagdividends','lagSPvol','lagtrade','lagamihud',"lagreturns","lagreturns2","lagreturns3",'lagcoeff2',"lagcoeff3",'laginterest2','laginterest3','lagvolatility2','lagvolatility3','lagSPprice2','lagSPprice3','lagdividends2','lagdividends3','lagSPvol2','lagSPvol3',"lagtrade2","lagtrade3",'lagamihud2','lagamihud3','laginterest4','lagvolatility4','lagcoeff4','lagtrade4', "lagamihud4", "lagreturns4","lagreturns5","laginterest5","lagvolatility5","lagcoeff5","lagtrade5","lagamihud5","lagreturns6","laginterest6","lagvolatility6","lagcoeff6","lagtrade6","lagamihud6","lagreturns7","laginterest7","lagvolatility7","lagcoeff7","lagtrade7","lagamihud7","lagreturns8","lagvolatility8","lagcoeff8","lagtrade8","lagamihud8","lagreturns9","lagvolatility9","lagcoeff9","lagtrade9","lagamihud9","lagreturns10","lagvolatility10","lagcoeff10","lagtrade10","lagamihud10","lagreturns11","lagvolatility11","lagcoeff11","lagtrade11","lagamihud11","lagreturns12","lagvolatility12","lagcoeff12","lagtrade12","lagamihud12","lagSPprice4","lagSPvol4","lagdividends4","lagSPprice5","lagSPvol5","lagdividends5","lagSPprice6","lagSPvol6","lagdividends6","lagSPprice7","lagSPvol7","lagdividends7","lagSPprice8","lagSPvol8","lagdividends8","lagSPprice9","lagSPvol9","lagdividends9","lagSPprice10","lagSPvol10","lagdividends10","lagSPprice11","lagSPvol11","lagdividends11","lagSPprice12","lagSPvol12","lagdividends12","laginterest8","laginterest9","laginterest10","laginterest11","laginterest12"])
        Y_test=test_data1.drop(columns=["coeff","interest","SPprice","SPvol","dividends","volatility","trade","amihud",'laginterest','lagvolatility','lagcoeff','lagSPprice','lagdividends','lagSPvol','lagtrade','lagamihud',"lagreturns","lagreturns2","lagreturns3",'lagcoeff2',"lagcoeff3",'laginterest2','laginterest3','lagvolatility2','lagvolatility3','lagSPprice2','lagSPprice3','lagdividends2','lagdividends3','lagSPvol2','lagSPvol3',"lagtrade2","lagtrade3",'lagamihud2','lagamihud3','laginterest4','lagvolatility4','lagcoeff4','lagtrade4', "lagamihud4", "lagreturns4","lagreturns5","laginterest5","lagvolatility5","lagcoeff5","lagtrade5","lagamihud5","lagreturns6","laginterest6","lagvolatility6","lagcoeff6","lagtrade6","lagamihud6","lagreturns7","laginterest7","lagvolatility7","lagcoeff7","lagtrade7","lagamihud7","lagreturns8","lagvolatility8","lagcoeff8","lagtrade8","lagamihud8","lagreturns9","lagvolatility9","lagcoeff9","lagtrade9","lagamihud9","lagreturns10","lagvolatility10","lagcoeff10","lagtrade10","lagamihud10","lagreturns11","lagvolatility11","lagcoeff11","lagtrade11","lagamihud11","lagreturns12","lagvolatility12","lagcoeff12","lagtrade12","lagamihud12","lagSPprice4","lagSPvol4","lagdividends4","lagSPprice5","lagSPvol5","lagdividends5","lagSPprice6","lagSPvol6","lagdividends6","lagSPprice7","lagSPvol7","lagdividends7","lagSPprice8","lagSPvol8","lagdividends8","lagSPprice9","lagSPvol9","lagdividends9","lagSPprice10","lagSPvol10","lagdividends10","lagSPprice11","lagSPvol11","lagdividends11","lagSPprice12","lagSPvol12","lagdividends12","laginterest8","laginterest9","laginterest10","laginterest11","laginterest12"])

        return X_train, X_test, Y_train, Y_test


    X_train1 = split(0.5)[0]
    X_test1 = split(0.5)[1]
    Y_train1 = split(0.5)[2]
    Y_test1 = split(0.5)[3]

    X_train2 = split(0.625)[0]
    X_test2 = split(0.625)[1]
    Y_train2 = split(0.625)[2]
    Y_test2 = split(0.625)[3]

    X_train3 = split(0.75)[0]
    X_test3 = split(0.75)[1]
    Y_train3 = split(0.75)[2]
    Y_test3 = split(0.75)[3]

    X_train4 = split(0.875)[0]
    X_test4 = split(0.875)[1]
    Y_train4 = split(0.875)[2]
    Y_test4 = split(0.875)[3]

    X_train5 = split(1)[0]
    X_test5 = split(1)[1]
    Y_train5 = split(1)[2]
    Y_test5 = split(1)[3]

    #Fixed effects
    exog_vars = params

    exog1 = sm.tools.tools.add_constant(X_train1[exog_vars])
    endog1 = Y_train1['returns']
    model_fe1 = PanelOLS(endog1, exog1, entity_effects = True) 
    fe_res1 = model_fe1.fit()

    exog2 = sm.tools.tools.add_constant(X_train2[exog_vars])
    endog2 = Y_train2['returns']
    model_fe2 = PanelOLS(endog2, exog2, entity_effects = True) 
    fe_res2 = model_fe2.fit() 

    exog3 = sm.tools.tools.add_constant(X_train3[exog_vars])
    endog3 = Y_train3['returns']
    model_fe3 = PanelOLS(endog3, exog3, entity_effects = True) 
    fe_res3 = model_fe3.fit() 

    exog4 = sm.tools.tools.add_constant(X_train4[exog_vars])
    endog4 = Y_train4['returns']
    model_fe4 = PanelOLS(endog4, exog4, entity_effects = True) 
    fe_res4 = model_fe4.fit() 

    exog5 = sm.tools.tools.add_constant(X_train5[exog_vars])
    endog5 = Y_train5['returns']
    model_fe5 = PanelOLS(endog5, exog5, entity_effects = True) 
    fe_res5 = model_fe5.fit() 

    rsquare = (fe_res1.rsquared + fe_res2.rsquared + fe_res3.rsquared + fe_res4.rsquared + fe_res5.rsquared)/5
    
    adj_rsquared1 = 1 - (((1-fe_res1.rsquared)*(fe_res1.nobs-1))/(fe_res1.nobs-len(exog_vars)-1))
    adj_rsquared2 = 1 - (((1-fe_res2.rsquared)*(fe_res2.nobs-1))/(fe_res2.nobs-len(exog_vars)-1))
    adj_rsquared3 = 1 - (((1-fe_res3.rsquared)*(fe_res3.nobs-1))/(fe_res3.nobs-len(exog_vars)-1))
    adj_rsquared4 = 1 - (((1-fe_res4.rsquared)*(fe_res4.nobs-1))/(fe_res4.nobs-len(exog_vars)-1))
    adj_rsquared5 = 1 - (((1-fe_res5.rsquared)*(fe_res5.nobs-1))/(fe_res5.nobs-len(exog_vars)-1))
    
    adj_rsquare = (adj_rsquared1 + adj_rsquared2 + adj_rsquared3 + adj_rsquared4 + adj_rsquared5)/5
    
    AIC1 = -2*fe_res1.loglik+2*len(exog_vars)
    AIC2 = -2*fe_res2.loglik+2*len(exog_vars)
    AIC3 = -2*fe_res3.loglik+2*len(exog_vars)
    AIC4 = -2*fe_res4.loglik+2*len(exog_vars)
    AIC5 = -2*fe_res5.loglik+2*len(exog_vars)
    
    AIC = (AIC1 + AIC2 + AIC3 + AIC4 + AIC5)/5


    #Predictions
    exoge1 = sm.tools.tools.add_constant(X_test1[exog_vars])
    predictions1 = fe_res1.predict(exoge1)
    joint1 = pd.merge(left=predictions1, right=Y_test1, how="left", left_on=["id","stamp"], right_on=["id","stamp"])
    joint_outliers1 = joint1
    
    test_rmse1 = np.sqrt(mean_squared_error(joint1["returns"], predictions1))
    test_mae1 = mean_absolute_error(joint1["returns"], predictions1)
    
    q_low1 = joint1["predictions"].quantile(0.01)
    q_hi1  = joint1["predictions"].quantile(0.99)
    joint1 = joint1[(joint1["predictions"] < q_hi1) & (joint1["predictions"] > q_low1)]
    q_low1 = joint1["returns"].quantile(0.01)
    q_hi1  = joint1["returns"].quantile(0.99)
    joint1 = joint1[(joint1["returns"] < q_hi1) & (joint1["returns"] > q_low1)]

    exoge2 = sm.tools.tools.add_constant(X_test2[exog_vars])
    predictions2 = fe_res2.predict(exoge2)
    joint2 = pd.merge(left=predictions2, right=Y_test2, how="left", left_on=["id","stamp"], right_on=["id","stamp"])
    joint_outliers2 = joint2
    
    test_rmse2 = np.sqrt(mean_squared_error(joint2["returns"], predictions2))
    test_mae2 = mean_absolute_error(joint2["returns"], predictions2)
    
    q_low2 = joint2["predictions"].quantile(0.01)
    q_hi2  = joint2["predictions"].quantile(0.99)
    joint2 = joint2[(joint2["predictions"] < q_hi2) & (joint2["predictions"] > q_low2)]
    q_low2 = joint2["returns"].quantile(0.01)
    q_hi2  = joint2["returns"].quantile(0.99)
    joint2 = joint2[(joint2["returns"] < q_hi2) & (joint2["returns"] > q_low2)]

    exoge3 = sm.tools.tools.add_constant(X_test3[exog_vars])
    predictions3 = fe_res3.predict(exoge3)
    joint3 = pd.merge(left=predictions3, right=Y_test3, how="left", left_on=["id","stamp"], right_on=["id","stamp"])
    joint_outliers3 = joint3
    
    test_rmse3 = np.sqrt(mean_squared_error(joint3["returns"], predictions3))
    test_mae3 = mean_absolute_error(joint3["returns"], predictions3)
    
    q_low3 = joint3["predictions"].quantile(0.01)
    q_hi3  = joint3["predictions"].quantile(0.99)
    joint3 = joint3[(joint3["predictions"] < q_hi3) & (joint3["predictions"] > q_low3)]
    q_low3 = joint3["returns"].quantile(0.01)
    q_hi3  = joint3["returns"].quantile(0.99)
    joint3 = joint3[(joint3["returns"] < q_hi3) & (joint3["returns"] > q_low3)]

    exoge4 = sm.tools.tools.add_constant(X_test4[exog_vars])
    predictions4 = fe_res4.predict(exoge4)
    joint4 = pd.merge(left=predictions4, right=Y_test4, how="left", left_on=["id","stamp"], right_on=["id","stamp"])
    joint_outliers4 = joint4
    
    test_rmse4 = np.sqrt(mean_squared_error(joint4["returns"], predictions4))
    test_mae4 = mean_absolute_error(joint4["returns"], predictions4)
    
    q_low4 = joint4["predictions"].quantile(0.01)
    q_hi4  = joint4["predictions"].quantile(0.99)
    joint4 = joint4[(joint4["predictions"] < q_hi4) & (joint4["predictions"] > q_low4)]
    q_low4 = joint4["returns"].quantile(0.01)
    q_hi4  = joint4["returns"].quantile(0.99)
    joint4 = joint4[(joint4["returns"] < q_hi4) & (joint4["returns"] > q_low4)]

    exoge5 = sm.tools.tools.add_constant(X_test5[exog_vars])
    predictions5 = fe_res5.predict(exoge5)
    joint5 = pd.merge(left=predictions5, right=Y_test5, how="left", left_on=["id","stamp"], right_on=["id","stamp"])
    joint_outliers5 = joint5
    
    test_rmse5 = np.sqrt(mean_squared_error(joint5["returns"], predictions5))
    test_mae5 = mean_absolute_error(joint5["returns"], predictions5)
    
    q_low5 = joint5["predictions"].quantile(0.01)
    q_hi5  = joint5["predictions"].quantile(0.99)
    joint5 = joint5[(joint5["predictions"] < q_hi5) & (joint5["predictions"] > q_low5)]
    q_low5 = joint5["returns"].quantile(0.01)
    q_hi5  = joint5["returns"].quantile(0.99)
    joint5 = joint5[(joint5["returns"] < q_hi5) & (joint5["returns"] > q_low5)]
    
    rmse = (test_rmse1 + test_rmse2 + test_rmse3 + test_rmse4 + test_rmse5)/5
    mae = (test_mae1 + test_mae2 + test_mae3 + test_mae4 + test_mae5)/5
    
    fig = figure()
    plt.subplots(1,2, figsize=(20,9))
    plt.subplot(1, 2, 1)
    plt.gca().set_title('Predictions without outliers')
    sns.regplot(joint1["returns"], joint1["predictions"], line_kws={"color":"purple", "lw":6})
    sns.regplot(joint2["returns"], joint2["predictions"], line_kws={"color":"red", "lw":6})
    sns.regplot(joint3["returns"], joint3["predictions"], line_kws={"color":"yellow", "lw":6})
    sns.regplot(joint4["returns"], joint4["predictions"], line_kws={"color":"pink", "lw":6})
    sns.regplot(joint5["returns"], joint5["predictions"], line_kws={"color":"green", "lw":6})
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    
    plt.subplot(1, 2, 2)
    plt.gca().set_title('Predictions with outliers')
    sns.regplot(joint_outliers1["returns"], joint_outliers1["predictions"], line_kws={"color":"purple", "lw":6})
    sns.regplot(joint_outliers2["returns"], joint_outliers2["predictions"], line_kws={"color":"red", "lw":6})
    sns.regplot(joint_outliers3["returns"], joint_outliers3["predictions"], line_kws={"color":"yellow", "lw":6})
    sns.regplot(joint_outliers4["returns"], joint_outliers4["predictions"], line_kws={"color":"pink", "lw":6})
    sns.regplot(joint_outliers5["returns"], joint_outliers5["predictions"], line_kws={"color":"green", "lw":6})
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.savefig('regsplit.png')
    
    
    
    return (rsquare, adj_rsquare, AIC, rmse, mae, fig)


# In[16]:


#Test a model
model1 = modeling(params = ["lagcoeff", "lagdividends", "lagSPprice", "lagSPvol", "laginterest","lagtrade","volatility","amihud"])


# In[17]:


#Model results
model1


# In[34]:


#Model 12 lags
splity = (julia.index[julia['id'] == "SDTH"])
training_data = julia.loc[0:splity[0]]
test_data = julia.loc[splity[0]:]

#List of dates
stamp_list = sorted(list(set(julia.stamp)))
date = (stamp_list[int(len(stamp_list)*0.8)]).strftime('%Y-%m-%d')

training_data = julia.loc[0:pooo[0]]
training_data = training_data[training_data["stamp"] < pd.to_datetime(date)].copy()
training_data = training_data.set_index(["id", "stamp"])

test_data = julia.loc[pooo[0]:]
test_data = test_data[test_data["stamp"] >= pd.to_datetime(date)].copy()
test_data = test_data.set_index(["id", "stamp"])

X_train =training_data.drop(columns=['returns'])
X_test =test_data.drop(columns=['returns'])
Y_train=training_data.drop(columns=["coeff","interest","SPprice","SPvol","dividends","volatility","trade","amihud",'laginterest','lagvolatility','lagcoeff','lagSPprice','lagdividends','lagSPvol','lagtrade','lagamihud',"lagreturns","lagreturns2","lagreturns3",'lagcoeff2',"lagcoeff3",'laginterest2','laginterest3','lagvolatility2','lagvolatility3','lagSPprice2','lagSPprice3','lagdividends2','lagdividends3','lagSPvol2','lagSPvol3',"lagtrade2","lagtrade3",'lagamihud2','lagamihud3','laginterest4','lagvolatility4','lagcoeff4','lagtrade4', "lagamihud4", "lagreturns4","lagreturns5","laginterest5","lagvolatility5","lagcoeff5","lagtrade5","lagamihud5","lagreturns6","laginterest6","lagvolatility6","lagcoeff6","lagtrade6","lagamihud6","lagreturns7","laginterest7","lagvolatility7","lagcoeff7","lagtrade7","lagamihud7","lagreturns8","lagvolatility8","lagcoeff8","lagtrade8","lagamihud8","lagreturns9","lagvolatility9","lagcoeff9","lagtrade9","lagamihud9","lagreturns10","lagvolatility10","lagcoeff10","lagtrade10","lagamihud10","lagreturns11","lagvolatility11","lagcoeff11","lagtrade11","lagamihud11","lagreturns12","lagvolatility12","lagcoeff12","lagtrade12","lagamihud12","lagSPprice4","lagSPvol4","lagdividends4","lagSPprice5","lagSPvol5","lagdividends5","lagSPprice6","lagSPvol6","lagdividends6","lagSPprice7","lagSPvol7","lagdividends7","lagSPprice8","lagSPvol8","lagdividends8","lagSPprice9","lagSPvol9","lagdividends9","lagSPprice10","lagSPvol10","lagdividends10","lagSPprice11","lagSPvol11","lagdividends11","lagSPprice12","lagSPvol12","lagdividends12","laginterest8","laginterest9","laginterest10","laginterest11","laginterest12"])
Y_test=test_data.drop(columns=["coeff","interest","SPprice","SPvol","dividends","volatility","trade","amihud",'laginterest','lagvolatility','lagcoeff','lagSPprice','lagdividends','lagSPvol','lagtrade','lagamihud',"lagreturns","lagreturns2","lagreturns3",'lagcoeff2',"lagcoeff3",'laginterest2','laginterest3','lagvolatility2','lagvolatility3','lagSPprice2','lagSPprice3','lagdividends2','lagdividends3','lagSPvol2','lagSPvol3',"lagtrade2","lagtrade3",'lagamihud2','lagamihud3','laginterest4','lagvolatility4','lagcoeff4','lagtrade4', "lagamihud4", "lagreturns4","lagreturns5","laginterest5","lagvolatility5","lagcoeff5","lagtrade5","lagamihud5","lagreturns6","laginterest6","lagvolatility6","lagcoeff6","lagtrade6","lagamihud6","lagreturns7","laginterest7","lagvolatility7","lagcoeff7","lagtrade7","lagamihud7","lagreturns8","lagvolatility8","lagcoeff8","lagtrade8","lagamihud8","lagreturns9","lagvolatility9","lagcoeff9","lagtrade9","lagamihud9","lagreturns10","lagvolatility10","lagcoeff10","lagtrade10","lagamihud10","lagreturns11","lagvolatility11","lagcoeff11","lagtrade11","lagamihud11","lagreturns12","lagvolatility12","lagcoeff12","lagtrade12","lagamihud12","lagSPprice4","lagSPvol4","lagdividends4","lagSPprice5","lagSPvol5","lagdividends5","lagSPprice6","lagSPvol6","lagdividends6","lagSPprice7","lagSPvol7","lagdividends7","lagSPprice8","lagSPvol8","lagdividends8","lagSPprice9","lagSPvol9","lagdividends9","lagSPprice10","lagSPvol10","lagdividends10","lagSPprice11","lagSPvol11","lagdividends11","lagSPprice12","lagSPvol12","lagdividends12","laginterest8","laginterest9","laginterest10","laginterest11","laginterest12"])

exog_vars12 = ['lagcoeff','lagcoeff2',"lagcoeff3","lagcoeff10","lagreturns","lagreturns2","lagreturns3","lagreturns4","lagreturns5","lagreturns6","lagreturns7","lagreturns8","lagreturns9","lagreturns10","lagreturns11","lagreturns12","SPprice",'lagSPprice3',"lagSPprice7","lagSPvol5","volatility",'lagvolatility','lagvolatility2','lagvolatility3','lagvolatility4',"lagvolatility5","lagvolatility6","amihud",'lagamihud',"lagamihud3","lagamihud7","lagamihud8",'lagtrade',"lagtrade7","lagtrade9","lagtrade12",'laginterest3', "laginterest12","dividends"]
exog12 = sm.tools.tools.add_constant(X_train[exog_vars12])
endog12 = Y_train['returns']
model_fe12 = PanelOLS(endog12, exog12, entity_effects = True) 
fe_res12 = model_fe12.fit()
print(fe_res12)


# In[25]:


#Model statistics
R_square_adjusted12 = 1 - (((1-fe_res12.rsquared)*(fe_res12.nobs-1))/(fe_res12.nobs-len(exog_vars12)-1))
AIC12 = -2*fe_res12.loglik+2*len(exog_vars12)


# In[44]:


#Predictions
exoge12 = sm.tools.tools.add_constant(X_test[exog_vars12])
predictions12 = fe_res12.predict(exoge12)
joint_outliers12 = pd.merge(left=predictions12, right=Y_test, how="left", left_on=["id","stamp"], right_on=["id","stamp"])
joint12 = pd.merge(left=predictions12, right=Y_test, how="left", left_on=["id","stamp"], right_on=["id","stamp"])
q_low = joint12["predictions"].quantile(0.01)
q_hi  = joint12["predictions"].quantile(0.99)
joint12 = joint12[(joint12["predictions"] < q_hi) & (joint12["predictions"] > q_low)]
q_low = joint12["returns"].quantile(0.01)
q_hi  = joint12["returns"].quantile(0.99)
joint12 = joint12[(joint12["returns"] < q_hi) & (joint12["returns"] > q_low)]


#create scatterplot with regression line and confidence interval lines
plt.subplots(1,2, figsize=(20,9))
plt.subplot(1, 2, 1)
plt.gca().set_title('Predictions without outliers')
sns.regplot(joint12["returns"], joint12["predictions"], line_kws={"color":"purple"})
plt.xlabel("True Values")
plt.ylabel("Predictions")

plt.subplot(1, 2, 2)
plt.gca().set_title('Predictions with outliers')
sns.regplot(joint_outliers12["returns"], joint_outliers12["predictions"], line_kws={"color":"purple"})
plt.xlabel("True Values")
plt.ylabel("Predictions")

plt.savefig('reg12.png')


# In[28]:


#Errors
test_mse12 = mean_squared_error(joint_outliers12["returns"], joint_outliers12["predictions"])
test_rmse12 = np.sqrt(test_mse12)
test_mae12 = mean_absolute_error(joint_outliers12["returns"], joint_outliers12["predictions"])


# In[35]:


#Model 12 lags (forecasting)
splity = (julia.index[julia['id'] == "SDTH"])
training_data = julia.loc[0:splity[0]]
test_data = julia.loc[splity[0]:]

#List of dates
stamp_list = sorted(list(set(julia.stamp)))
date = (stamp_list[int(len(stamp_list)*0.8)]).strftime('%Y-%m-%d')

training_data = julia.loc[0:pooo[0]]
training_data = training_data[training_data["stamp"] < pd.to_datetime(date)].copy()
training_data = training_data.set_index(["id", "stamp"])

test_data = julia.loc[pooo[0]:]
test_data = test_data[test_data["stamp"] >= pd.to_datetime(date)].copy()
test_data = test_data.set_index(["id", "stamp"])

X_train =training_data.drop(columns=['returns'])
X_test =test_data.drop(columns=['returns'])
Y_train=training_data.drop(columns=["coeff","interest","SPprice","SPvol","dividends","volatility","trade","amihud",'laginterest','lagvolatility','lagcoeff','lagSPprice','lagdividends','lagSPvol','lagtrade','lagamihud',"lagreturns","lagreturns2","lagreturns3",'lagcoeff2',"lagcoeff3",'laginterest2','laginterest3','lagvolatility2','lagvolatility3','lagSPprice2','lagSPprice3','lagdividends2','lagdividends3','lagSPvol2','lagSPvol3',"lagtrade2","lagtrade3",'lagamihud2','lagamihud3','laginterest4','lagvolatility4','lagcoeff4','lagtrade4', "lagamihud4", "lagreturns4","lagreturns5","laginterest5","lagvolatility5","lagcoeff5","lagtrade5","lagamihud5","lagreturns6","laginterest6","lagvolatility6","lagcoeff6","lagtrade6","lagamihud6","lagreturns7","laginterest7","lagvolatility7","lagcoeff7","lagtrade7","lagamihud7","lagreturns8","lagvolatility8","lagcoeff8","lagtrade8","lagamihud8","lagreturns9","lagvolatility9","lagcoeff9","lagtrade9","lagamihud9","lagreturns10","lagvolatility10","lagcoeff10","lagtrade10","lagamihud10","lagreturns11","lagvolatility11","lagcoeff11","lagtrade11","lagamihud11","lagreturns12","lagvolatility12","lagcoeff12","lagtrade12","lagamihud12","lagSPprice4","lagSPvol4","lagdividends4","lagSPprice5","lagSPvol5","lagdividends5","lagSPprice6","lagSPvol6","lagdividends6","lagSPprice7","lagSPvol7","lagdividends7","lagSPprice8","lagSPvol8","lagdividends8","lagSPprice9","lagSPvol9","lagdividends9","lagSPprice10","lagSPvol10","lagdividends10","lagSPprice11","lagSPvol11","lagdividends11","lagSPprice12","lagSPvol12","lagdividends12","laginterest8","laginterest9","laginterest10","laginterest11","laginterest12"])
Y_test=test_data.drop(columns=["coeff","interest","SPprice","SPvol","dividends","volatility","trade","amihud",'laginterest','lagvolatility','lagcoeff','lagSPprice','lagdividends','lagSPvol','lagtrade','lagamihud',"lagreturns","lagreturns2","lagreturns3",'lagcoeff2',"lagcoeff3",'laginterest2','laginterest3','lagvolatility2','lagvolatility3','lagSPprice2','lagSPprice3','lagdividends2','lagdividends3','lagSPvol2','lagSPvol3',"lagtrade2","lagtrade3",'lagamihud2','lagamihud3','laginterest4','lagvolatility4','lagcoeff4','lagtrade4', "lagamihud4", "lagreturns4","lagreturns5","laginterest5","lagvolatility5","lagcoeff5","lagtrade5","lagamihud5","lagreturns6","laginterest6","lagvolatility6","lagcoeff6","lagtrade6","lagamihud6","lagreturns7","laginterest7","lagvolatility7","lagcoeff7","lagtrade7","lagamihud7","lagreturns8","lagvolatility8","lagcoeff8","lagtrade8","lagamihud8","lagreturns9","lagvolatility9","lagcoeff9","lagtrade9","lagamihud9","lagreturns10","lagvolatility10","lagcoeff10","lagtrade10","lagamihud10","lagreturns11","lagvolatility11","lagcoeff11","lagtrade11","lagamihud11","lagreturns12","lagvolatility12","lagcoeff12","lagtrade12","lagamihud12","lagSPprice4","lagSPvol4","lagdividends4","lagSPprice5","lagSPvol5","lagdividends5","lagSPprice6","lagSPvol6","lagdividends6","lagSPprice7","lagSPvol7","lagdividends7","lagSPprice8","lagSPvol8","lagdividends8","lagSPprice9","lagSPvol9","lagdividends9","lagSPprice10","lagSPvol10","lagdividends10","lagSPprice11","lagSPvol11","lagdividends11","lagSPprice12","lagSPvol12","lagdividends12","laginterest8","laginterest9","laginterest10","laginterest11","laginterest12"])

exog_vars12f = ['lagcoeff','lagcoeff2',"lagcoeff3","lagcoeff10","lagreturns","lagreturns2","lagreturns3","lagreturns4","lagreturns5","lagreturns6","lagreturns7","lagreturns8","lagreturns9","lagreturns10","lagreturns11","lagreturns12",'lagSPprice3',"lagSPprice7","lagSPvol5",'lagvolatility','lagvolatility2','lagvolatility3','lagvolatility4',"lagvolatility5","lagvolatility6",'lagamihud',"lagamihud3","lagamihud7","lagamihud8",'lagtrade',"lagtrade7","lagtrade9","lagtrade12",'laginterest3', "laginterest12"]
exog12f = sm.tools.tools.add_constant(X_train[exog_vars12f])
endog12f = Y_train['returns']
model_fe12f = PanelOLS(endog12f, exog12f, entity_effects = True) 
fe_res12f = model_fe12f.fit()
print(fe_res12f)


# In[30]:


#Model statistics
R_square_adjusted12f = 1 - (((1-fe_res12f.rsquared)*(fe_res12f.nobs-1))/(fe_res12f.nobs-len(exog_vars12f)-1))
AIC12f = -2*fe_res12f.loglik+2*len(exog_vars12f)


# In[38]:


#Predictions
exoge12f = sm.tools.tools.add_constant(X_test[exog_vars12f])
predictions12f = fe_res12f.predict(exoge12f)
joint_outliers12f = pd.merge(left=predictions12f, right=Y_test, how="left", left_on=["id","stamp"], right_on=["id","stamp"])
joint12f = pd.merge(left=predictions12f, right=Y_test, how="left", left_on=["id","stamp"], right_on=["id","stamp"])
q_low = joint12f["predictions"].quantile(0.01)
q_hi  = joint12f["predictions"].quantile(0.99)
joint12f = joint12f[(joint12f["predictions"] < q_hi) & (joint12f["predictions"] > q_low)]
q_low = joint12f["returns"].quantile(0.01)
q_hi  = joint12f["returns"].quantile(0.99)
joint12f = joint12f[(joint12f["returns"] < q_hi) & (joint12f["returns"] > q_low)]


#create scatterplot with regression line and confidence interval lines
plt.subplots(1,2, figsize=(20,9))
plt.subplot(1, 2, 1)
plt.gca().set_title('Predictions without outliers')
sns.regplot(joint12f["returns"], joint12f["predictions"], line_kws={"color":"purple"})
plt.xlabel("True Values")
plt.ylabel("Predictions")

plt.subplot(1, 2, 2)
plt.gca().set_title('Predictions with outliers')
sns.regplot(joint_outliers12f["returns"], joint_outliers12f["predictions"], line_kws={"color":"purple"})
plt.xlabel("True Values")
plt.ylabel("Predictions")

plt.savefig('reg12f.png')


# In[33]:


#Errors
test_mse12f = mean_squared_error(joint_outliers12f["returns"], joint_outliers12f["predictions"])
test_rmse12f = np.sqrt(test_mse12f)
test_mae12f = mean_absolute_error(joint_outliers12f["returns"], joint_outliers12f["predictions"])


# In[36]:


#Model 6 lags
splity = (julia.index[julia['id'] == "SDTH"])
training_data = julia.loc[0:splity[0]]
test_data = julia.loc[splity[0]:]

#List of dates
stamp_list = sorted(list(set(julia.stamp)))
date = (stamp_list[int(len(stamp_list)*0.8)]).strftime('%Y-%m-%d')

training_data = julia.loc[0:pooo[0]]
training_data = training_data[training_data["stamp"] < pd.to_datetime(date)].copy()
training_data = training_data.set_index(["id", "stamp"])

test_data = julia.loc[pooo[0]:]
test_data = test_data[test_data["stamp"] >= pd.to_datetime(date)].copy()
test_data = test_data.set_index(["id", "stamp"])

X_train =training_data.drop(columns=['returns'])
X_test =test_data.drop(columns=['returns'])
Y_train=training_data.drop(columns=["coeff","interest","SPprice","SPvol","dividends","volatility","trade","amihud",'laginterest','lagvolatility','lagcoeff','lagSPprice','lagdividends','lagSPvol','lagtrade','lagamihud',"lagreturns","lagreturns2","lagreturns3",'lagcoeff2',"lagcoeff3",'laginterest2','laginterest3','lagvolatility2','lagvolatility3','lagSPprice2','lagSPprice3','lagdividends2','lagdividends3','lagSPvol2','lagSPvol3',"lagtrade2","lagtrade3",'lagamihud2','lagamihud3','laginterest4','lagvolatility4','lagcoeff4','lagtrade4', "lagamihud4", "lagreturns4","lagreturns5","laginterest5","lagvolatility5","lagcoeff5","lagtrade5","lagamihud5","lagreturns6","laginterest6","lagvolatility6","lagcoeff6","lagtrade6","lagamihud6","lagreturns7","laginterest7","lagvolatility7","lagcoeff7","lagtrade7","lagamihud7","lagreturns8","lagvolatility8","lagcoeff8","lagtrade8","lagamihud8","lagreturns9","lagvolatility9","lagcoeff9","lagtrade9","lagamihud9","lagreturns10","lagvolatility10","lagcoeff10","lagtrade10","lagamihud10","lagreturns11","lagvolatility11","lagcoeff11","lagtrade11","lagamihud11","lagreturns12","lagvolatility12","lagcoeff12","lagtrade12","lagamihud12","lagSPprice4","lagSPvol4","lagdividends4","lagSPprice5","lagSPvol5","lagdividends5","lagSPprice6","lagSPvol6","lagdividends6","lagSPprice7","lagSPvol7","lagdividends7","lagSPprice8","lagSPvol8","lagdividends8","lagSPprice9","lagSPvol9","lagdividends9","lagSPprice10","lagSPvol10","lagdividends10","lagSPprice11","lagSPvol11","lagdividends11","lagSPprice12","lagSPvol12","lagdividends12","laginterest8","laginterest9","laginterest10","laginterest11","laginterest12"])
Y_test=test_data.drop(columns=["coeff","interest","SPprice","SPvol","dividends","volatility","trade","amihud",'laginterest','lagvolatility','lagcoeff','lagSPprice','lagdividends','lagSPvol','lagtrade','lagamihud',"lagreturns","lagreturns2","lagreturns3",'lagcoeff2',"lagcoeff3",'laginterest2','laginterest3','lagvolatility2','lagvolatility3','lagSPprice2','lagSPprice3','lagdividends2','lagdividends3','lagSPvol2','lagSPvol3',"lagtrade2","lagtrade3",'lagamihud2','lagamihud3','laginterest4','lagvolatility4','lagcoeff4','lagtrade4', "lagamihud4", "lagreturns4","lagreturns5","laginterest5","lagvolatility5","lagcoeff5","lagtrade5","lagamihud5","lagreturns6","laginterest6","lagvolatility6","lagcoeff6","lagtrade6","lagamihud6","lagreturns7","laginterest7","lagvolatility7","lagcoeff7","lagtrade7","lagamihud7","lagreturns8","lagvolatility8","lagcoeff8","lagtrade8","lagamihud8","lagreturns9","lagvolatility9","lagcoeff9","lagtrade9","lagamihud9","lagreturns10","lagvolatility10","lagcoeff10","lagtrade10","lagamihud10","lagreturns11","lagvolatility11","lagcoeff11","lagtrade11","lagamihud11","lagreturns12","lagvolatility12","lagcoeff12","lagtrade12","lagamihud12","lagSPprice4","lagSPvol4","lagdividends4","lagSPprice5","lagSPvol5","lagdividends5","lagSPprice6","lagSPvol6","lagdividends6","lagSPprice7","lagSPvol7","lagdividends7","lagSPprice8","lagSPvol8","lagdividends8","lagSPprice9","lagSPvol9","lagdividends9","lagSPprice10","lagSPvol10","lagdividends10","lagSPprice11","lagSPvol11","lagdividends11","lagSPprice12","lagSPvol12","lagdividends12","laginterest8","laginterest9","laginterest10","laginterest11","laginterest12"])

exog_vars6 = ['lagcoeff','lagcoeff2',"lagcoeff3",'lagcoeff4',"lagcoeff6","lagreturns","lagreturns2","lagreturns3","lagreturns4","lagreturns5","lagreturns6","SPprice",'lagSPprice2','lagSPprice3',"volatility",'lagvolatility','lagvolatility2','lagvolatility3','lagvolatility4',"lagvolatility5","lagvolatility6","amihud","lagamihud",'lagamihud2',"lagamihud3","lagamihud4","lagamihud5","lagamihud6","trade",'lagtrade','lagtrade4', "lagtrade5","lagtrade6","laginterest3","dividends"]
exog6 = sm.tools.tools.add_constant(X_train[exog_vars6])
endog6 = Y_train['returns']
model_fe6 = PanelOLS(endog6, exog6, entity_effects = True) 
fe_res6 = model_fe6.fit()
print(fe_res6)


# In[37]:


#Model statistics
R_square_adjusted6 = 1 - (((1-fe_res6.rsquared)*(fe_res6.nobs-1))/(fe_res6.nobs-len(exog_vars6)-1))
AIC6 = -2*fe_res6.loglik+2*len(exog_vars6)


# In[45]:


#Predictions
exoge6 = sm.tools.tools.add_constant(X_test[exog_vars6])
predictions6 = fe_res6.predict(exoge6)
joint_outliers6 = pd.merge(left=predictions6, right=Y_test, how="left", left_on=["id","stamp"], right_on=["id","stamp"])
joint6 = pd.merge(left=predictions6, right=Y_test, how="left", left_on=["id","stamp"], right_on=["id","stamp"])
q_low = joint6["predictions"].quantile(0.01)
q_hi  = joint6["predictions"].quantile(0.99)
joint6 = joint6[(joint6["predictions"] < q_hi) & (joint6["predictions"] > q_low)]
q_low = joint6["returns"].quantile(0.01)
q_hi  = joint6["returns"].quantile(0.99)
joint6 = joint6[(joint6["returns"] < q_hi) & (joint6["returns"] > q_low)]


#create scatterplot with regression line and confidence interval lines
plt.subplots(1,2, figsize=(20,9))
plt.subplot(1, 2, 1)
plt.gca().set_title('Predictions without outliers')
sns.regplot(joint6["returns"], joint6["predictions"], line_kws={"color":"purple"})
plt.xlabel("True Values")
plt.ylabel("Predictions")

plt.subplot(1, 2, 2)
plt.gca().set_title('Predictions with outliers')
sns.regplot(joint_outliers6["returns"], joint_outliers6["predictions"], line_kws={"color":"purple"})
plt.xlabel("True Values")
plt.ylabel("Predictions")

plt.savefig('reg6.png')


# In[40]:


#Errors
test_mse6 = mean_squared_error(joint_outliers6["returns"], joint_outliers6["predictions"])
test_rmse6 = np.sqrt(test_mse6)
test_mae6 = mean_absolute_error(joint_outliers6["returns"], joint_outliers6["predictions"])


# In[41]:


#Model 3 lags
splity = (julia.index[julia['id'] == "SDTH"])
training_data = julia.loc[0:splity[0]]
test_data = julia.loc[splity[0]:]

#List of dates
stamp_list = sorted(list(set(julia.stamp)))
date = (stamp_list[int(len(stamp_list)*0.8)]).strftime('%Y-%m-%d')

training_data = julia.loc[0:pooo[0]]
training_data = training_data[training_data["stamp"] < pd.to_datetime(date)].copy()
training_data = training_data.set_index(["id", "stamp"])

test_data = julia.loc[pooo[0]:]
test_data = test_data[test_data["stamp"] >= pd.to_datetime(date)].copy()
test_data = test_data.set_index(["id", "stamp"])

X_train =training_data.drop(columns=['returns'])
X_test =test_data.drop(columns=['returns'])
Y_train=training_data.drop(columns=["coeff","interest","SPprice","SPvol","dividends","volatility","trade","amihud",'laginterest','lagvolatility','lagcoeff','lagSPprice','lagdividends','lagSPvol','lagtrade','lagamihud',"lagreturns","lagreturns2","lagreturns3",'lagcoeff2',"lagcoeff3",'laginterest2','laginterest3','lagvolatility2','lagvolatility3','lagSPprice2','lagSPprice3','lagdividends2','lagdividends3','lagSPvol2','lagSPvol3',"lagtrade2","lagtrade3",'lagamihud2','lagamihud3'])
Y_test=test_data.drop(columns=["coeff","interest","SPprice","SPvol","dividends","volatility","trade","amihud",'laginterest','lagvolatility','lagcoeff','lagSPprice','lagdividends','lagSPvol','lagtrade','lagamihud',"lagreturns","lagreturns2","lagreturns3",'lagcoeff2',"lagcoeff3",'laginterest2','laginterest3','lagvolatility2','lagvolatility3','lagSPprice2','lagSPprice3','lagdividends2','lagdividends3','lagSPvol2','lagSPvol3',"lagtrade2","lagtrade3",'lagamihud2','lagamihud3'])

exog_vars3 = ['lagcoeff','lagcoeff2',"lagcoeff3","lagreturns","lagreturns2","lagreturns3","SPprice",'lagSPprice2',"volatility",'lagvolatility','lagvolatility2',"lagvolatility3","amihud","lagamihud",'lagamihud2',"lagamihud3",'lagtrade',"lagtrade3","dividends"]
exog3 = sm.tools.tools.add_constant(X_train[exog_vars3])
endog3 = Y_train['returns']
model_fe3 = PanelOLS(endog3, exog3, entity_effects = True) 
fe_res3 = model_fe3.fit()
print(fe_res3)


# In[42]:


#Model statistics
R_square_adjusted3 = 1 - (((1-fe_res3.rsquared)*(fe_res3.nobs-1))/(fe_res3.nobs-len(exog_vars3)-1))
AIC3 = -2*fe_res3.loglik+2*len(exog_vars3)


# In[43]:


#Predictions
exoge3 = sm.tools.tools.add_constant(X_test[exog_vars3])
predictions3 = fe_res3.predict(exoge3)
joint_outliers3 = pd.merge(left=predictions3, right=Y_test, how="left", left_on=["id","stamp"], right_on=["id","stamp"])
joint3 = pd.merge(left=predictions3, right=Y_test, how="left", left_on=["id","stamp"], right_on=["id","stamp"])
q_low = joint3["predictions"].quantile(0.01)
q_hi  = joint3["predictions"].quantile(0.99)
joint3 = joint3[(joint3["predictions"] < q_hi) & (joint3["predictions"] > q_low)]
q_low = joint3["returns"].quantile(0.01)
q_hi  = joint3["returns"].quantile(0.99)
joint3 = joint3[(joint3["returns"] < q_hi) & (joint3["returns"] > q_low)]


#create scatterplot with regression line and confidence interval lines
plt.subplots(1,2, figsize=(20,9))
plt.subplot(1, 2, 1)
plt.gca().set_title('Predictions without outliers')
sns.regplot(joint3["returns"], joint3["predictions"], line_kws={"color":"purple"})
plt.xlabel("True Values")
plt.ylabel("Predictions")

plt.subplot(1, 2, 2)
plt.gca().set_title('Predictions with outliers')
sns.regplot(joint_outliers3["returns"], joint_outliers3["predictions"], line_kws={"color":"purple"})
plt.xlabel("True Values")
plt.ylabel("Predictions")

plt.savefig('reg3.png')


# In[46]:


#Errors
test_mse3 = mean_squared_error(joint_outliers3["returns"], joint_outliers3["predictions"])
test_rmse3 = np.sqrt(test_mse3)
test_mae3 = mean_absolute_error(joint_outliers3["returns"], joint_outliers3["predictions"])


# In[47]:


#Model 1 lag
splity = (julia.index[julia['id'] == "SDTH"])
training_data = julia.loc[0:splity[0]]
test_data = julia.loc[splity[0]:]

#List of dates
stamp_list = sorted(list(set(julia.stamp)))
date = (stamp_list[int(len(stamp_list)*0.8)]).strftime('%Y-%m-%d')

training_data = julia.loc[0:pooo[0]]
training_data = training_data[training_data["stamp"] < pd.to_datetime(date)].copy()
training_data = training_data.set_index(["id", "stamp"])

test_data = julia.loc[pooo[0]:]
test_data = test_data[test_data["stamp"] >= pd.to_datetime(date)].copy()
test_data = test_data.set_index(["id", "stamp"])

X_train =training_data.drop(columns=['returns'])
X_test =test_data.drop(columns=['returns'])
Y_train=training_data.drop(columns=["coeff","interest","SPprice","SPvol","dividends","volatility","trade","amihud",'laginterest','lagvolatility','lagcoeff','lagSPprice','lagdividends','lagSPvol','lagtrade','lagamihud',"lagreturns","lagreturns2","lagreturns3",'lagcoeff2',"lagcoeff3",'laginterest2','laginterest3','lagvolatility2','lagvolatility3','lagSPprice2','lagSPprice3','lagdividends2','lagdividends3','lagSPvol2','lagSPvol3',"lagtrade2","lagtrade3",'lagamihud2','lagamihud3'])
Y_test=test_data.drop(columns=["coeff","interest","SPprice","SPvol","dividends","volatility","trade","amihud",'laginterest','lagvolatility','lagcoeff','lagSPprice','lagdividends','lagSPvol','lagtrade','lagamihud',"lagreturns","lagreturns2","lagreturns3",'lagcoeff2',"lagcoeff3",'laginterest2','laginterest3','lagvolatility2','lagvolatility3','lagSPprice2','lagSPprice3','lagdividends2','lagdividends3','lagSPvol2','lagSPvol3',"lagtrade2","lagtrade3",'lagamihud2','lagamihud3'])

exog_vars = ['lagcoeff',"lagreturns","SPprice","volatility",'lagvolatility',"amihud","lagamihud",'lagtrade',"dividends"]
exog = sm.tools.tools.add_constant(X_train[exog_vars])
endog = Y_train['returns']
model_fe = PanelOLS(endog, exog, entity_effects = True) 
fe_res = model_fe.fit()
print(fe_res)


# In[48]:


#Model statistics
R_square_adjusted = 1 - (((1-fe_res.rsquared)*(fe_res.nobs-1))/(fe_res.nobs-len(exog_vars)-1))
AIC = -2*fe_res.loglik+2*len(exog_vars)


# In[49]:


#Predictions
exoge = sm.tools.tools.add_constant(X_test[exog_vars])
predictions = fe_res.predict(exoge)
joint_outliers = pd.merge(left=predictions, right=Y_test, how="left", left_on=["id","stamp"], right_on=["id","stamp"])
joint = pd.merge(left=predictions, right=Y_test, how="left", left_on=["id","stamp"], right_on=["id","stamp"])
q_low = joint["predictions"].quantile(0.01)
q_hi  = joint["predictions"].quantile(0.99)
joint = joint[(joint["predictions"] < q_hi) & (joint["predictions"] > q_low)]
q_low = joint["returns"].quantile(0.01)
q_hi  = joint["returns"].quantile(0.99)
joint = joint[(joint["returns"] < q_hi) & (joint["returns"] > q_low)]


#create scatterplot with regression line and confidence interval lines
plt.subplots(1,2, figsize=(20,9))
plt.subplot(1, 2, 1)
plt.gca().set_title('Predictions without outliers')
sns.regplot(joint["returns"], joint["predictions"], line_kws={"color":"purple"})
plt.xlabel("True Values")
plt.ylabel("Predictions")

plt.subplot(1, 2, 2)
plt.gca().set_title('Predictions with outliers')
sns.regplot(joint_outliers["returns"], joint_outliers["predictions"], line_kws={"color":"purple"})
plt.xlabel("True Values")
plt.ylabel("Predictions")

plt.savefig('reg.png')


# In[50]:


#Errors
test_mse = mean_squared_error(joint_outliers["returns"], joint_outliers["predictions"])
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(joint_outliers["returns"], joint_outliers["predictions"])


# In[51]:


comparison = compare({"12 month": fe_res12, "6 month": fe_res6, "3 month": fe_res3, "1 month":fe_res}, precision = ("pvalues"), stars = True)


# In[52]:


print(comparison)

