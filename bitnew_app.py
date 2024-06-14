
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objs as go

# Function to fetch historical BTC data
def fetch_btc_data(ticker='BTC-USD', period='70d'):
    btc_data = yf.download(ticker, period=period, interval='1d')
    return btc_data

# Load the LSTM model
model = load_model('btcprice_prediction_model.h5')

# Load the MinMaxScaler
scaler = MinMaxScaler()

def app():
    st.title('GM Analytics Next Day Bitcoin Price Prediction AI')
    st.write('This app Auto collects the latest 60 days of Bitcoin close prices from Yahoo Finance and uses an LSTM model to predict the next day\'s close price.')

    if st.button('Fetch Latest Data and Predict'):
        btc_data = fetch_btc_data()

        if not btc_data.empty:
            btc_data.dropna(inplace=True)  # Remove rows with missing data
            st.write('Fetched data:')
            st.dataframe(btc_data.tail(60))

            if len(btc_data) < 60:
                st.error('Insufficient data. Ensure that you have at least 60 days of data.')
                return

            btc_close_prices = btc_data[['Close']]

            try:
                last_60_days = btc_close_prices[-60:].values
                last_60_days_scaled = scaler.fit_transform(last_60_days)

                # Prepare the data for prediction
                X_test = []
                X_test.append(last_60_days_scaled)
                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                # Predict the next day's price
                pred_price_scaled = model.predict(X_test)
                pred_price = scaler.inverse_transform(pred_price_scaled)

                st.success(f'Predicted Bitcoin Price for the Next Day: ${pred_price[0, 0]:.2f}')

                # Placeholder for loading the data (to be replaced with actual data loading)
                data = pd.read_csv('C:/Users/User/Documents/NorthCentralUniversity/ModelDeployment/bit/data.csv')
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)

                # Scale the data
                scaled_data = scaler.fit_transform(data[['Close']])

                # Split the data into training and test sets
                training_data_len = int(len(data) * 0.8)
                train_data = scaled_data[:training_data_len]
                test_data = scaled_data[training_data_len - 60:]

                # Prepare the input sequences and target values for testing
                x_test, y_test = [], []
                for i in range(60, len(test_data)):
                    x_test.append(test_data[i - 60:i, 0])
                    y_test.append(test_data[i, 0])

                # Convert the input sequences and target values to NumPy arrays
                x_test = np.array(x_test)
                y_test = np.array(y_test)

                # Reshape the input data to match the model's expected input shape
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

                # Use the model to make predictions on the test data
                predictions = model.predict(x_test)

                # Invert the scaling for the predictions and actual values
                predictions = scaler.inverse_transform(predictions)
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

                # Visualize the trained model with tested data using Plotly
                train = data[:training_data_len]
                valid = data[training_data_len:]
                valid['Predictions'] = predictions

                # Creating the Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train.index, y=train['Close'], name='Train'))
                fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], name='Val'))
                fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], name='Predictions'))

                # Set Layout
                fig.update_layout(
                    title='LSTM Model - Crypto:Bitcoin - Trained Model',
                    xaxis=dict(title='Date', showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(title='Bitcoin Close Price USD ($)', showgrid=True, gridcolor='lightgray'),
                    legend=dict(x=0.01, y=0.99, bordercolor='black', borderwidth=1),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    showlegend=True,
                )
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f'An error occurred during prediction: {str(e)}')
        else:
            st.error('Failed to fetch data from Yahoo Finance.')

if __name__ == '__main__':
    app()

# Open a terminal or command prompt, navigate to the directory containing btc_prediction_app.py, and run:
# streamlit run btc_prediction_app.py

# streamlit run "C:/Users/User/Documents/NorthCentralUniversity/ModelDeployment/bit/autobtcprediction_app.py"
