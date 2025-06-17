import pandas as pd

brka_d = pd.read_csv("data/processed/brka_d_ret.csv", parse_dates=True, index_col=0)
print(brka_d.head())