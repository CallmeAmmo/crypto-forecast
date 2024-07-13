import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import datetime as dt
from env_var import env

# Fetch historical data
def fetch_data(symbol, comparison_symbol, limit, to_date=None):
    base_url = 'https://min-api.cryptocompare.com/data/v2/histohour'
    if to_date is not None:
        toTs = int(dt.datetime.strptime(to_date, '%Y-%m-%d').timestamp())
        url = f'{base_url}?fsym={symbol}&tsym={comparison_symbol}&limit={limit}&toTs={toTs}'
    else:
        url = f'{base_url}?fsym={symbol}&tsym={comparison_symbol}&limit={limit}'
    
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data['Data']['Data'])

# Fetch data for the last 2000 hours
df = fetch_data('BTC', 'USD', 2000, to_date=env.last_date)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# Select features and target
features = ['open', 'high', 'low', 'close', 'volumefrom', 'volumeto']
target = 'close'

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# Save the scaler
joblib.dump(scaler, 'scaler.save')

# Prepare the data for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i])
        y.append(data[i, 3])  # target is 'close'
    return np.array(X), np.array(y)

time_step = 100  # Use past 100 hours to predict the next hour
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], len(features))

# Split into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, len(features))))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with validation split
model.fit(X_train, y_train, batch_size=1, epochs=10, validation_split=0.2)

# Save the trained model
model.save('lstm_model.h5')

print("Model and scaler saved successfully.")
