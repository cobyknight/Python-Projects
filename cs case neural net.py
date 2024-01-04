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

item_names = ['Operation Bravo Case', 'Desert Eagle | Golden Koi (Minimal Wear)', 'AK-47 | Fire Serpent (Field-Tested)', 'AWP | Graphite (Minimal Wear)', 'P2000 | Ocean Foam (Minimal Wear)', 'P90 | Emerald Dragon (Field-Tested)', 'M4A1-S | Bright Water (Field-Tested)']
data_frames = []
for item_name in item_names:
    url = f"https://steamcommunity.com/market/pricehistory/?country=US&currency=1&appid=730&market_hash_name={item_name}"
    headers = {"Cookie": "timezoneOffset=-18000,0; browserid=2771453880978516196; strInventoryLastContext=730_2; steamLoginSecure=76561198139263088%7C%7CeyAidHlwIjogIkpXVCIsICJhbGciOiAiRWREU0EiIH0.eyAiaXNzIjogInI6MEQxOF8yMkU5RUY4RF8zMDE1NiIsICJzdWIiOiAiNzY1NjExOTgxMzkyNjMwODgiLCAiYXVkIjogWyAid2ViIiBdLCAiZXhwIjogMTY5MjEzMjE0MywgIm5iZiI6IDE2ODM0MDQ3NDcsICJpYXQiOiAxNjkyMDQ0NzQ3LCAianRpIjogIjEyMzZfMjJGREI0QkJfMzUxNUYiLCAib2F0IjogMTY5MDQwMTE4NywgInJ0X2V4cCI6IDE3MDg1OTM2NjksICJwZXIiOiAwLCAiaXBfc3ViamVjdCI6ICI5OS4xMS4yMy44MiIsICJpcF9jb25maXJtZXIiOiAiMTY2LjE5NC4xNDMuNTIiIH0.KDsvS-ocgnZ0lI63JZklSaT57DGjnp2Ti5cfrTdjR2tPj11JFSr0UHrsr5fts3_hZWPO32HpCmhdFqc9BVKICA; sessionid=a9b32561d830733509c805eb"}
    print(item_name)
    response = requests.get(url, headers=headers)
    data = response.json()
    df = pd.DataFrame.from_dict(data)
    df = df.drop(['success', 'price_prefix', 'price_suffix'], axis=1)
    df[['date', f'{item_name}_price', f'{item_name}_volume']] = df['prices'].apply(
        lambda x: pd.Series(x)
    )
    df = df.drop('prices', axis=1)
    data_frames.append(df)
    print(df)

# Initialize the merged data frame with the first data frame in the list
merged_df = data_frames[0]

# Merge the remaining data frames on the 'date' column
for df in data_frames[1:]:
    merged_df = merged_df.merge(df, on='date', how='inner')

# Rename the columns of the merged data frame
merged_df.columns = ['date'] + [f'{item_name}_price' for item_name in item_names] + [f'{item_name}_volume' for item_name in item_names]

# Display the merged data frame
print(merged_df)

# Normalize the data
scaler = MinMaxScaler()
for item_name in item_names:
    merged_df[f'{item_name}_volume'] = scaler.fit_transform(
        np.expand_dims(merged_df[f'{item_name}_volume'].values, axis=1)
    )
    merged_df[f'{item_name}_price'] = scaler.fit_transform(
        np.expand_dims(merged_df[f'{item_name}_price'].values, axis=1)
    )

print(merged_df)

#define the steps (7 days = 1 week)
N_STEPS = 7

#prepare the data for machine learning
def PrepareData(days):
    df = merged_df.copy()
    df['future'] = df[f'{item_names[0]}_price'].shift(-days)
    last_sequence = np.array(
        df[
            [f'{item_name}_price' for item_name in item_names]
            + [f'{item_name}_volume' for item_name in item_names]
        ].tail(days)
    )
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=N_STEPS)

    for entry, target in zip(
        df[
            [f'{item_name}_price' for item_name in item_names]
            + [f'{item_name}_volume' for item_name in item_names]
        ].values,
        df['future'].values,
    ):
        sequences.append(entry)
        if len(sequences) == N_STEPS:
            sequence_data.append([np.array(sequences), target])

    last_sequence = (
        list([s[: len([f'{item_name}_price' for item_name in item_names] + [f'{item_name}_volume' for item_name in item_names])] for s in sequences])
        + list(last_sequence)
    )
    last_sequence = np.array(last_sequence).astype(np.float32)

    print(df)

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
    model.add(
        LSTM(60, return_sequences=True, input_shape=(N_STEPS, len(item_names) * 2))
    )
    model.add(Dropout(0.3))
    model.add(LSTM(120, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(20))
    model.add(Dense(1))

    BATCH_SIZE = 8
    EPOCHS = 80

    model.compile(loss='mean_squared_error', optimizer='adam')

    #trains the model
    model.fit(
        x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1
    )

    model.summary()

    return model

#lookup steps

LOOKUP_STEPS = [30, 60, 90]

# GET PREDICTIONS
predictions = []


for step in LOOKUP_STEPS:
    df, last_sequence, x_train, y_train = PrepareData(step)
    x_train = x_train[:, :, :len(item_names) * 2 * 2].astype(np.float32)

    model = GetTrainedModel(x_train, y_train)

    # Prepare the input for prediction
    last_sequence = last_sequence[-N_STEPS:]
    last_sequence = np.expand_dims(last_sequence, axis=0)  # Ensure the same number of features
    # Ensure the same number of features

    # Make the prediction
    prediction = model.predict(last_sequence)

    # Inverse transform the prediction
    predicted_price = scaler.inverse_transform(prediction)[0][0]

    predictions.append(round(float(predicted_price), 2))

if bool(predictions) == True and len(predictions) > 0:
    predictions_list = [str(d)+'$' for d in predictions]
    predictions_str = ', '.join(predictions_list)
    message = f'CSGO prediction for upcoming 30, 60, and 90 days ({predictions_str})'

    print(message)


# Create a new data frame with the actual price values
df = merged_df.copy()
df[f'{item_names[0]}_price'] = scaler.inverse_transform(np.expand_dims(df[f'{item_names[0]}_price'].values, axis=1))

# Plot the actual price history
plt.style.use(style='ggplot')
plt.figure(figsize=(16,10))
plt.plot(df[f'{item_names[0]}_price'][-3100:])
plt.xlabel("days")
plt.ylabel("price")
plt.legend([f'Actual price for item price'])

# Plot the predicted prices
x = [len(df) + i for i in LOOKUP_STEPS]
y = predictions
plt.plot(x, y, color='blue', marker='o')

# Show the plot
plt.show()
