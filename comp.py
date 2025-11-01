import pandas as pd

df = pd.read_csv('data/raw/drugLibTrain_raw.csv')
head = df.head
columns = df.columns
missing = df.isna()
dfWithoutNA = df.dropna()

if df.isnull().any().any():
    print(True)
else:
    print(False)
if dfWithoutNA.isnull().any().any():
    print(True)
else:
    print(False)

df.to_csv('drugLibTrain_clean.csv')