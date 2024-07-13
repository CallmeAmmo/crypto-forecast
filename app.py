import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from infer import fetch_data, predict_next_hours

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
    df = fetch_data('BTC', 'USD', 2000, to_date=None)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Select features
    features = ['open', 'high', 'low', 'close', 'volumefrom', 'volumeto']

    # Load the saved model
    saved_model = load_model('lstm_model.h5')

    # Load the saved scaler
    scaler = joblib.load('scaler.save')

    # Get the last 'time_step' rows from the data for prediction
    time_step = 100  # This should be the same as used during training
    last_data = df[features].values[-time_step:]
    start_time = df.index[-1]

    # Predict the closing price for the next 24 hours
    predictions_df = predict_next_hours(saved_model, scaler, last_data, 24, time_step, features, start_time)

    # Display predictions
    st.write(predictions_df)

    # Calculate and display metrics
    true_values = df['close'].values[-24:]
    pred_values = predictions_df['predicted_close'].values
    mape = mean_absolute_percentage_error(true_values, pred_values)
    mse = mean_squared_error(true_values, pred_values)
    st.write(f"**MAPE:** {mape:.2%}")
    st.write(f"**MSE:** {mse:.2f}")

    # Plotting
    fig = plot_predictions(df.index, df['close'].values, predictions_df['time'], predictions_df['predicted_close'])
    st.pyplot(fig)

if __name__ == "__main__":
    main()
