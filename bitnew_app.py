# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:40:24 2024

@author: User
"""

# btc_prediction_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the LSTM model
#model = load_model('btcprice_prediction_model.h5')

model = load_model ('btcprice_prediction_model.h5')
# Load the MinMaxScaler
scaler = MinMaxScaler()

def app():
    st.title('GM Analytics Next Day Bitcoin Price Prediction AI')

    # Input features for prediction
    sequence_length = 60  # Set to the sequence length expected by your model
    st.text(f'Sequence length is set to {sequence_length} (model expects this for Higher Accuracy)')

    user_input = st.text_area(
        f'Enter the last {sequence_length} days closing prices (one value per Column). Get Btc prices here: https://finance.yahoo.com/quote/BTC-USD/history/',
        value=''
    )

    # Convert user input to a list of floats
    try:
        user_input_list = [float(value.strip()) for value in user_input.split('\n') if value.strip()]
        if len(user_input_list) != sequence_length:
            st.error(f'Please enter exactly {sequence_length} numerical values.')
            return
    except ValueError:
        st.error('Invalid input. Please enter numeric values, one per line.')
        return

    if st.button('Predict'):
        try:
            # Convert user input to a NumPy array and reshape for the model
            user_input_array = np.array(user_input_list).reshape(-1, 1)

            # Normalize the user input
            user_input_scaled = scaler.fit_transform(user_input_array)

            # Reshape to match model input
            user_input_scaled = user_input_scaled.reshape(1, sequence_length, 1)

            # Predict the next day's price
            prediction_scaled = model.predict(user_input_scaled)

            # Invert the scaling for the prediction
            prediction = scaler.inverse_transform(prediction_scaled)

            st.success('Predicted Bitcoin Price for the Next Day: ${:.2f}'.format(prediction[0, 0]))

            # Placeholder for loading the data (to be replaced with actual data loading)
            data = pd.read_csv('data.csv')
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)

            # Scale the data
            scaled_data = scaler.fit_transform(data[['Close']])

            # Split the data into training and test sets
            training_data_len = int(len(data) * 0.8)
            train_data = scaled_data[:training_data_len]
            test_data = scaled_data[training_data_len - sequence_length:]

            # Prepare the input sequences and target values for testing
            x_test, y_test = [], []
            for i in range(sequence_length, len(test_data)):
                x_test.append(test_data[i - sequence_length:i, 0])
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

if __name__ == '__main__':
    app()

# Instructions for running the app:
# Open a terminal or command prompt, navigate to the directory containing btc_prediction_app.py, and run:
    
    
# streamlit run "C:/Users/User/Documents/NorthCentralUniversity/ModelDeployment/bit/bitnew_app.py"
