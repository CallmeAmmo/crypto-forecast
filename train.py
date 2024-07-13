# import numpy as np
# import pandas as pd
# import requests
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import joblib
# import datetime as dt
# from env_var import env

# # Fetch historical data
# def fetch_data(symbol, comparison_symbol, limit, to_date=None):
#     base_url = 'https://min-api.cryptocompare.com/data/v2/histohour'
#     if to_date is not None:
#         toTs = int(dt.datetime.strptime(to_date, '%Y-%m-%d').timestamp())
#         url = f'{base_url}?fsym={symbol}&tsym={comparison_symbol}&limit={limit}&toTs={toTs}'
#     else:
#         url = f'{base_url}?fsym={symbol}&tsym={comparison_symbol}&limit={limit}'
    
#     response = requests.get(url)
#     data = response.json()
#     df = pd.DataFrame(data['Data']['Data'])
#     df =df[:-1]

#     df['time'] = pd.to_datetime(df['time'], unit='s')
#     df.set_index('time', inplace=True)
    
#     return df

# # Fetch data for the last 2000 hours
# df = fetch_data('BTC', 'USD', 2000, to_date=env.last_date)
# df.to_csv('data.csv')

# # Select features and target
# features = env.features
# target = 'close'

# # Scale the data
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(df[features])

# # Save the scaler
# joblib.dump(scaler, 'scaler.save')

# # Prepare the data for LSTM
# def create_dataset(data, time_step=1):
#     X, y = [], []
#     for i in range(time_step, len(data)):
#         X.append(data[i-time_step:i])
#         y.append(data[i, 3])  # target is 'close'
#     return np.array(X), np.array(y)

# time_step = env.lag  # Use past 100 hours to predict the next hour
# X, y = create_dataset(scaled_data, time_step)
# X = X.reshape(X.shape[0], X.shape[1], len(features))

# # Split into training and test sets
# test_size = env.test_hours
# X_train, X_test = X[: -1*test_size], X[-1*test_size:]
# y_train, y_test = y[: -1*test_size], y[-1*test_size:]

# # Build the LSTM model
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, len(features))))
# model.add(LSTM(units=50, return_sequences=False))
# model.add(Dense(units=25))
# model.add(Dense(units=1))

# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model with validation split
# model.fit(X_train, y_train, batch_size=1, epochs=10, validation_split=0.3)

# # Save the trained model
# model.save('lstm_model.h5')

# print("Model and scaler saved successfully.")




import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import datetime as dt
from env_var import env

def fetch_data(symbol, comparison_symbol, limit, to_date=None):
    base_url = 'https://min-api.cryptocompare.com/data/v2/histohour'
    if to_date is not None:
        toTs = int(dt.datetime.strptime(to_date, '%Y-%m-%d').timestamp())
        url = f'{base_url}?fsym={symbol}&tsym={comparison_symbol}&limit={limit}&toTs={toTs}'
    else:
        url = f'{base_url}?fsym={symbol}&tsym={comparison_symbol}&limit={limit}'
    
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['Data']['Data'])
    df = df[:-1]

    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    return df

def prepare_data(df, features, time_step):
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # Save the scaler
    joblib.dump(scaler, 'scaler.save')
    
    # Prepare the data for LSTM
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], len(features))
    
    return X, y, scaler

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i])
        y.append(data[i, 3])  # target is 'close'
    return np.array(X), np.array(y)

def split_data(X, y, test_size):
    X_train, X_test = X[: -1*test_size], X[-1*test_size:]
    y_train, y_test = y[: -1*test_size], y[-1*test_size:]
    return X_train, X_test, y_train, y_test

def build_model(time_step, feature_count):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, feature_count)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=1, validation_split=0.3):
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    return model

def save_model(model, scaler, model_filename='lstm_model.h5', scaler_filename='scaler.save'):
    model.save(model_filename)
    joblib.dump(scaler, scaler_filename)
    print("Model and scaler saved successfully.")

def main():
    # Fetch data
    df = fetch_data('BTC', 'USD', 2000, to_date=env.last_date)
    df.to_csv('data.csv')
    
    # Prepare data
    features = env.features
    time_step = env.lag
    X, y, scaler = prepare_data(df, features, time_step)
    
    # Split data
    test_size = env.test_hours
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)
    
    # Build and train model
    model = build_model(time_step, len(features))
    model = train_model(model, X_train, y_train, epochs=10, batch_size=1, validation_split=0.3)
    
    # Save model and scaler
    save_model(model, scaler)

if __name__ == "__main__":
    main()
