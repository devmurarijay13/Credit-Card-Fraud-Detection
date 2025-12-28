# import pandas as pd
# testing sample
# import pickle
# import os

# from sklearn.preprocessing import RobustScaler


# df = pd.read_csv("notebook/data/creditcard.csv")

# scaler = RobustScaler()

# df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
# df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))

# os.makedirs('artifacts',exist_ok=True)
# df.drop(columns=['Class'],inplace=True)

# with open('artifacts/features.pkl','wb') as f:
#     pickle.dump(df.columns.to_list(),f)
# print(df.sample(1).to_dict())



import requests

url = "https://jay-devmurari-13-credit-card-fraud-detection.hf.space/predict"

payload = {
  "data": {
    "Time": -0.2909808620871956,
    "V1": 0.684770277350612,
    "V2": -0.502631084538354,
    "V3": -0.288273698507291,
    "V4": 0.482213083496719,
    "V5": 0.464848663167097,
    "V6": 0.865015800777525,
    "V7": 0.413812157097448,
    "V8": 0.1833705448045,
    "V9": -0.325981085969492,
    "V10": -0.36445540226348,
    "V11": 0.86343107785291,
    "V12": 1.00364680250616,
    "V13": 0.625258930939542,
    "V14": 0.524409408969064,
    "V15": 1.80683709507824,
    "V16": -1.24158588695811,
    "V17": 0.925294053779662,
    "V18": -2.86050984025876,
    "V19": -1.51533689698107,
    "V20": 0.234843853220817,
    "V21": -0.12753796223119,
    "V22": -0.705408720112382,
    "V23": 0.0787276861016296,
    "V24": -0.94662112857882,
    "V25": -0.086044450857824,
    "V26": 0.242267272597258,
    "V27": -0.028638789756604,
    "V28": 0.0332039765085073,
    "Amount": 2.766016907706281
  }
}


r = requests.post(url, json=payload)
print(r.json())
