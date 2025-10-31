import pandas as pd 

data=pd.read_csv("data/drugsComTrain_raw.csv")
# print(data.head)
data

d1=data["drugName"].unique()
print(d1)
ratings = pd.read_csv("data/drug_ratings.csv")
d2=ratings["Drug Name"] 

sum=0
for d1 in d2:
    sum = sum+ 1
    
print(sum)
