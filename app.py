import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from env_var import env


# Function to calculate the average change
def calculate_average_percentage_change(data, column):
    percentage_changes = data[column].pct_change().dropna()
    return percentage_changes.mean()


# Function to predict the closing price for the next N hours
def predict_next_hours(model, scaler, last_data, num_hours, time_step, features, start_time):
    predictions = []
    timestamps = []
    data = last_data.copy()
    current_time = start_time

    # Calculate average changes for high and low adjustments
    df_last = pd.DataFrame(data, columns=features)
    avg_change_high = calculate_average_percentage_change(df_last, 'high')
    avg_change_low = calculate_average_percentage_change(df_last, 'low')

    avg_change_volumefrom = calculate_average_percentage_change(df_last, 'volumefrom')
    avg_change_volumeto = calculate_average_percentage_change(df_last, 'volumeto')

    for _ in range(num_hours):
        scaled_data = scaler.transform(data[-time_step:])
        X_input = scaled_data.reshape(1, time_step, len(features))
        pred = model.predict(X_input)
        pred_inversed = scaler.inverse_transform(np.concatenate((pred, np.zeros((pred.shape[0], len(features) - 1))), axis=1))[:, 0]
        predictions.append(pred_inversed[0])
        timestamps.append(current_time)
        current_time += pd.Timedelta(hours=1)

        # Create a new row with the predicted close value and the necessary adjustments
        new_row = data[-1].copy()
        new_row[0] = new_row[3]  # open price of new row = close price of previous row
        new_row[3] = pred_inversed[0]  # close price of new row = predicted close

        # Introduce random positive or negative change for high and low prices
        high_change =  avg_change_high * new_row[1] * np.random.choice([-1, 0, 1])
        low_change =  avg_change_low * new_row[2] * np.random.choice([-1, 0, 1, 2])

        # Introduce random positive or negative change for volumefrom and volumeto
        volumefrom_change =  avg_change_volumefrom * new_row[4] * np.random.choice([-1, 1])
        volumeto_change = avg_change_volumeto * new_row[5] * np.random.choice([-1, 1])

        # features = ['open', 'high', 'low', 'close', 'volumefrom', 'volumeto']

        new_row[1] = new_row[1] + high_change  # high price of new row
        new_row[2] = new_row[2] + low_change   # low price of new row

        new_row[4] = new_row[4] + volumefrom_change
        new_row[5] = new_row[5] + volumeto_change

        # Ensure that the high is greater than or equal to both the open and close
        new_row[1] = max(new_row[1], new_row[3], new_row[0])
        
        # Ensure that the low is less than or equal to both the open and close
        new_row[2] = min(new_row[2], new_row[3], new_row[0])

        # Store the new row

        data = np.vstack([data, new_row])
    
    # Use raw predictions without smoothing
    predictions_raw = np.array(predictions)
    
    return pd.DataFrame({'time': timestamps, 'predicted_close': predictions_raw})


def fetch_data():
    df = pd.read_csv('data.csv')

    # Preprocess the data
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    return df


# Function to plot historical and predicted prices
def plot_predictions(true_dates, true_values, pred_dates, pred_values):
    fig, ax = plt.subplots()
    ax.plot(true_dates, true_values, label='Historical Prices')
    ax.plot(pred_dates, pred_values, label='Predicted Prices', linestyle='--')
    ax.legend()
    ax.set_title("Crypto Price Prediction")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    plt.xticks(rotation=45)
    return fig

def main():
    st.title("Crypto Price Prediction")

    # Fetch data
    total_df = fetch_data()

    df =total_df.iloc[:-1* env.test_hours].copy()

    # Select features
    features = env.features

    # Load the saved model
    saved_model = load_model('lstm_model.h5')

    # Load the saved scaler
    scaler = joblib.load('scaler.save')

    # Get the last 'time_step' rows from the data for prediction
    time_step = env.lag  # This should be the same as used during training
    last_data = df[features].values[-time_step:]
    start_time = df.index[-1] + pd.Timedelta(hours=1)

    # Predict the closing price for the next 24 hours
    prediction_hours = env.prediction_hours
    predictions_df = predict_next_hours(saved_model, scaler, last_data, prediction_hours, time_step, features, start_time)

    # Display predictions
    test_hours = env.test_hours

    st_df = predictions_df[:test_hours].copy()
    st_df['actual'] = total_df[-1*test_hours:]['close'].values
    st.write(st_df)

    # Calculate and display metrics
    true_values = total_df['close'].values[-1*test_hours:]
    pred_values = predictions_df['predicted_close'].values[:test_hours]

    mape = mean_absolute_percentage_error(true_values, pred_values)
    mse = mean_absolute_error(true_values, pred_values)
    st.write(f"**MAPE:** {mape:.2%}")
    st.write(f"**MAE:** {mse:.2f}")

    # Plotting
    plt_df = total_df[-50:]
    fig = plot_predictions(plt_df.index, plt_df['close'].values, predictions_df['time'], predictions_df['predicted_close'])
    st.pyplot(fig)

if __name__ == "__main__":
    main()
    print('done')
