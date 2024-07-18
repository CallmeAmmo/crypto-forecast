import numpy as np
import pandas as pd
import joblib
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from env_var import env
import webbrowser
from threading import Timer

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
        high_change = avg_change_high * new_row[1] * np.random.choice([-1, 0, 1])
        low_change = avg_change_low * new_row[2] * np.random.choice([-1, 0, 1])

        # Introduce random positive or negative change for volumefrom and volumeto
        volumefrom_change = avg_change_volumefrom * new_row[4] * np.random.choice([-1, 1])
        volumeto_change = avg_change_volumeto * new_row[5] * np.random.choice([-1, 1])

        new_row[1] = new_row[1] + high_change  # high price of new row
        new_row[2] = new_row[2] + low_change   # low price of new row

        new_row[4] = new_row[4] + volumefrom_change
        new_row[5] = new_row[5] + volumeto_change

        new_row[1] = max(new_row[1], new_row[3], new_row[0])
        new_row[2] = min(new_row[2], new_row[3], new_row[0])

        data = np.vstack([data, new_row])
    
    predictions_raw = np.array(predictions)
    
    return pd.DataFrame({'time': timestamps, 'predicted_close': predictions_raw})

def fetch_data():
    df = pd.read_csv('data.csv')
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df

# Load the data
total_df = fetch_data()
df = total_df.iloc[:-1 * env.test_hours].copy()
features = env.features

saved_model = load_model('lstm_model.h5')
scaler = joblib.load('scaler.save')

time_step = env.lag
last_data = df[features].values[-time_step:]
start_time = df.index[-1] + pd.Timedelta(hours=1)

prediction_hours = env.prediction_hours
predictions_df = predict_next_hours(saved_model, scaler, last_data, prediction_hours, time_step, features, start_time)

print(predictions_df)
test_hours = env.test_hours
next_forecst = predictions_df[test_hours:].copy()

st_df = predictions_df[:test_hours].copy()
st_df['actual'] = total_df[-1 * test_hours:]['close'].values

true_values = total_df['close'].values[-1 * test_hours:]
pred_values = predictions_df['predicted_close'].values[:test_hours]
mape = mean_absolute_percentage_error(true_values, pred_values)
mae = mean_absolute_error(true_values, pred_values)

# Dash app
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Crypto Price Prediction'),

    html.Div(children=f"MAPE: {mape:.2%}"),
    html.Div(children=f"MAE: {mae:.2f}"),

    dcc.Graph(
        id='price-prediction-graph',
        figure={
            'data': [
                go.Scatter(
                    x=total_df.index[-200:],
                    y=total_df['close'].values[-200:],
                    mode='lines',
                    name='Historical Prices'
                ),
                go.Scatter(
                    x=predictions_df['time'],
                    y=predictions_df['predicted_close'],
                    mode='lines',
                    name='Predicted Prices'
                )
            ],
            'layout': {
                'title': 'Crypto Price Prediction',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Price'},
                'legend': {'x': 0, 'y': 1},
                'hovermode': 'closest'
            }
        }
    ),

    dash_table.DataTable(
        id='prediction-table',
        columns=[{"name": i, "id": i} for i in next_forecst.columns],
        data=next_forecst.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        }
    )
])

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run_server(debug=True)
