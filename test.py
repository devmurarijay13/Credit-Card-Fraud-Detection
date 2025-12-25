import pandas as pd
# testing sample
import pickle
import os

from sklearn.preprocessing import RobustScaler


df = pd.read_csv("notebook/data/creditcard.csv")

scaler = RobustScaler()

df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))

os.makedirs('artifacts',exist_ok=True)
df.drop(columns=['Class'])

with open('artifacts/features.pkl','wb') as f:
    pickle.dump(df.columns.to_list(),f)