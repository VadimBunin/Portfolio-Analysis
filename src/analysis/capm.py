import pandas as pd
from utilis import  risk_kit as erk

brka_d = pd.read_csv("data/processed/brka_d_ret.csv", parse_dates=True, index_col=0)
brka_m=brka_d.resample('ME').apply(erk.compound).to_period('M')
#brka_m.to_csv("data/clips/brka_m.csv") # for possible future use!
fff = erk.get_fff_returns()
print(f"explanatory variables, which is the Fama-French monthly returns data set \n{fff.tail()}")
import statsmodels.api as sm
import numpy as np
brka_excess = brka_m["1990":"2012-05"] - fff.loc["1990":"2012-05", ['RF']].values
mkt_excess = fff.loc["1990":"2012-05",['Mkt-RF']]
exp_var = mkt_excess.copy()
exp_var["Constant"] = 1
lm = sm.OLS(brka_excess, exp_var).fit()
print(lm.summary())