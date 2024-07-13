import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import load_model
import joblib
import datetime as dt
from env_var import env

# Function to predict the closing price for the next N hours
def predict_next_hours(model, scaler, last_data, num_hours, time_step, features, start_time):
    predictions = []
    timestamps = []
    data = last_data.copy()
    current_time = start_time

    for _ in range(num_hours):
        scaled_data = scaler.transform(data[-time_step:])
        X_input = scaled_data.reshape(1, time_step, len(features))
        pred = model.predict(X_input)
        pred_inversed = scaler.inverse_transform(np.concatenate((pred, np.zeros((pred.shape[0], len(features) - 1))), axis=1))[:, 0]
        predictions.append(pred_inversed[0])
        timestamps.append(current_time)
        current_time += pd.Timedelta(hours=1)

        # Add the predicted value as the next 'close' and shift the window
        new_row = np.append(data[-1, 1:], pred_inversed[0]).reshape(1, len(features))
        data = np.vstack([data, new_row])
    
    return pd.DataFrame({'time': timestamps, 'predicted_close': predictions})

# Load the saved model
saved_model = load_model('lstm_model.h5')

# Load the saved scaler
scaler = joblib.load('scaler.save')

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

# Select features
features = ['open', 'high', 'low', 'close', 'volumefrom', 'volumeto']

# Get the last 'time_step' rows from the data for prediction
time_step = 100  # This should be the same as used during training
last_data = df[features].values[-time_step:]
start_time = df.index[-1]

# Predict the closing price for the next 24 hours
predictions_df = predict_next_hours(saved_model, scaler, last_data, 24, time_step, features, start_time)
predictions_df.to_csv('predictions_1.csv', index=False)
print(predictions_df)
