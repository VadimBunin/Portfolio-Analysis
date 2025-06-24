import pandas as pd
from utilis import  risk_kit as erk
import statsmodels.api as sm
import numpy as np



brka_d = pd.read_csv("data/brka_d_ret.csv", parse_dates=True, index_col=0)
brka_m = brka_d.resample('ME').apply(erk.compound).to_period('M')
fff = erk.get_fff_returns()
brka_excess = brka_m["1990":"2012-05"] - fff.loc["1990":"2012-05", ['RF']].values
mkt_excess = fff.loc["1990":"2012-05",['Mkt-RF']]
exp_var = mkt_excess.copy()
exp_var["Value"] = fff.loc["1990":"2012-05",['HML']]
exp_var["Size"] = fff.loc["1990":"2012-05",['SMB']]



def regress(dependent_variable, explanatory_variables, alpha=True):
    """
    Runs a linear regression to decompose the dependent variable into the explanatory variables
    returns an object of type statsmodel's RegressionResults on which you can call
       .summary() to print a full summary
       .params for the coefficients
       .tvalues and .pvalues for the significance levels
       .rsquared_adj and .rsquared for quality of fit
    """
    if alpha:
        explanatory_variables = explanatory_variables.copy()
        explanatory_variables["Alpha"] = 1
        
    ls = sm.OLS(dependent_variable, explanatory_variables).fit()
    return ls

class LinReg:
    def __init__(self, y, x):
        self.dep_var = y
        self.exp_var = x

    def run_regression(self, alpha=1):
        X = self.exp_var.copy()
        if alpha == 1:
            X = sm.add_constant(X, has_constant='add')
        model = sm.OLS(self.dep_var, X).fit()
        return model
        
   
linreg = LinReg(brka_excess,exp_var)     
print(linreg.run_regression().summary())