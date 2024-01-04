import numpy as np
import time as tm
import datetime as dt
import tensorflow as tf

# Data preparation
from yahoo_fin import stock_info as yf
from sklearn.preprocessing import MinMaxScaler
from collections import deque

# AI
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Graphics library
import matplotlib.pyplot as plt

import requests
import json
import matplotlib.pyplot as plt
import pandas as pd
from pandas import json_normalize
import numpy as np
#https://steamcommunity.com/market/listings/730/AK-47%20%7C%20Fuel%20Injector%20%28Factory%20New%29
url = "https://steamcommunity.com/market/pricehistory/?country=US&currency=1&appid=730&market_hash_name=M4A1-S%20%7C%20Bright%20Water%20%28Field-Tested%29"
headers = {"Cookie": "timezoneOffset=-18000,0; browserid=2771453880978516196; strInventoryLastContext=730_2; sessionid=347c4260d49e167c8ded7514; steamCountry=US%7Cbe0ff3eda0f4c4e4911b6caee3ee9404; steamDidLoginRefresh=1693257063; steamLoginSecure=76561198139263088%7C%7CeyAidHlwIjogIkpXVCIsICJhbGciOiAiRWREU0EiIH0.eyAiaXNzIjogInI6MEQxOF8yMkU5RUY4RF8zMDE1NiIsICJzdWIiOiAiNzY1NjExOTgxMzkyNjMwODgiLCAiYXVkIjogWyAid2ViIiBdLCAiZXhwIjogMTY5MzM0MzY3NCwgIm5iZiI6IDE2ODQ2MTcwNjQsICJpYXQiOiAxNjkzMjU3MDY0LCAianRpIjogIjBEMTJfMjMwRURENjlfRDZEMjYiLCAib2F0IjogMTY5MDQwMTE4NywgInJ0X2V4cCI6IDE3MDg1OTM2NjksICJwZXIiOiAwLCAiaXBfc3ViamVjdCI6ICI5OS4xMS4yMy44MiIsICJpcF9jb25maXJtZXIiOiAiMTY2LjE5NC4xNDMuNTIiIH0.spdsNoShidFSSRZZl-czbh3yqDW0vVkI74Cvocf211JbuFIfb4nBnXN7IQ1uKAGnwjE6ZAFgYigc4iucXEUBDw"}
response = requests.get(url, headers=headers)
data = response.json()

ct = 1
yval = []
xval = []
for item in data.get('prices'):
    yval.append(item[1])
    xval.append(item[0])
    ct += 1
    
'''
plt.plot(xval, yval)

# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')

# giving a title to my graph

plt.title('CS item')

plt.xticks(np.arange(0, len(xval)+1,100))
# function to show the plot
#plt.show()
'''

init_df = pd.DataFrame.from_dict(data)
#init_df = init_df.iloc[:-150]
init_df = init_df.drop(['success', 'price_prefix', 'price_suffix'], axis=1)

init_df[['date', 'price', 'volume']] = init_df['prices'].apply(lambda x: pd.Series(x))
init_df = init_df.drop('prices', axis=1)
init_df['date2'] = init_df.index

#prints data
print(init_df)

#plots data
plt.style.use(style='ggplot')
plt.figure(figsize=(16,10))
plt.plot(init_df['price'][-3000:])
plt.xlabel("days")
plt.ylabel("price")
plt.legend([f'Actual price for item price'])
plt.show()

#normalize the data
scaler = MinMaxScaler()
init_df['volume'] = scaler.fit_transform(np.expand_dims(init_df['volume'].values, axis=1))
init_df['price'] = scaler.fit_transform(np.expand_dims(init_df['price'].values, axis=1))

#define the steps (7 days = 1 week)
N_STEPS = 7

#prepare the data for machine learning
def PrepareData(days):
    df = init_df.copy()
    df['future'] = df['price'].shift(-days)
    last_sequence = np.array(df[['price', 'volume']].tail(days))
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=N_STEPS)
    
    for entry, target in zip(df[['price', 'volume'] + ['date']].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == N_STEPS:
            sequence_data.append([np.array(sequences), target])
            
    last_sequence = list([s[:len(['price', 'volume'])] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    
    X, Y = [], []
    for seq, target in sequence_data:
      X.append(seq)
      Y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    return df, last_sequence, X, Y

def GetTrainedModel(x_train, y_train):
  #builds neural net layers
  model = Sequential()
  model.add(LSTM(60, return_sequences=True, input_shape=(N_STEPS, len(['price', 'volume']))))
  model.add(Dropout(0.3))
  model.add(LSTM(120, return_sequences=False))
  model.add(Dropout(0.3))
  model.add(Dense(20))
  model.add(Dense(1))

  BATCH_SIZE = 8
  EPOCHS = 80

  model.compile(loss='mean_squared_error', optimizer='adam')

  #trains the model
  model.fit(x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1)

  model.summary()

  return model

#lookup steps

LOOKUP_STEPS = [30, 60, 90]

# GET PREDICTIONS
predictions = []

for step in LOOKUP_STEPS:
  df, last_sequence, x_train, y_train = PrepareData(step)
  x_train = x_train[:, :, :len(['price', 'volume'])].astype(np.float32)

  model = GetTrainedModel(x_train, y_train)

  last_sequence = last_sequence[-N_STEPS:]
  last_sequence = np.expand_dims(last_sequence, axis=0)
  
  prediction = model.predict(last_sequence)
  predicted_price = scaler.inverse_transform(prediction)[0][0]

  predictions.append(round(float(predicted_price), 2))

if bool(predictions) == True and len(predictions) > 0:
  predictions_list = [str(d)+'$' for d in predictions]
  predictions_str = ', '.join(predictions_list)
  message = f'CSGO prediction for upcoming 30, 60, and 90 days ({predictions_str})'

  print(message)

# Create a new data frame with the actual price values
df = init_df.copy()
df['price'] = scaler.inverse_transform(np.expand_dims(df['price'].values, axis=1))

# Plot the actual price history
plt.style.use(style='ggplot')
plt.figure(figsize=(16,10))
plt.plot(df['price'][-3100:])
plt.xlabel("days")
plt.ylabel("price")
plt.legend([f'Actual price for item price'])

# Plot the predicted prices
x = [len(df) + i for i in LOOKUP_STEPS]
y = predictions
plt.plot(x, y, color='blue', marker='o')

# Show the plot
plt.show()