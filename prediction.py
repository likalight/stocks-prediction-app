import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

def prepare_input_data(stock_data, n_timesteps, scaler):
    """Prepare input data with the last n_timesteps of historical data for prediction."""
    input_data = stock_data.iloc[-n_timesteps:][['Open', 'High', 'Low', 'Close', 'Adj Close', 
                                                 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26']].values
    input_data_scaled = scaler.transform(input_data)  # Scale the input data
    input_data_scaled = input_data_scaled.reshape((1, n_timesteps, len(input_data[0])))  # Reshape for LSTM input
    return input_data_scaled

def predict_stock(ticker, model, stock_data, time_range, n_timesteps, scaler):
    """Predict stock prices for the selected time range and tweak predictions to fit historical values with more variation."""
    input_data_scaled = prepare_input_data(stock_data, n_timesteps, scaler)

    # Set days to predict based on time range
    days_to_predict = 7 if time_range == "1 Week" else 30 if time_range == "1 Month" else 180

    predictions = []
    last_known_date = stock_data['Date'].max()
    forecasted_dates = []
    last_known_close = stock_data['Close'].iloc[-1]  # Get the last known close price

    for i in range(days_to_predict):
        # Predict for the next day
        pred_value_scaled = model.predict(input_data_scaled)[0][0]
        predictions.append(pred_value_scaled)

        # Shift the input data (slide window over time)
        new_row = input_data_scaled[:, -1, :].copy()
        new_row[0, 3] = pred_value_scaled  # Update the 'Close' value (index 3)

        input_data_scaled = np.concatenate([input_data_scaled[:, 1:, :], new_row.reshape(1, 1, input_data_scaled.shape[2])], axis=1)
        forecasted_dates.append(last_known_date + pd.Timedelta(days=i + 1))

    predictions = np.array(predictions).reshape(-1, 1)  # Reshape predictions

    # Inverse scale predictions
    n_features = len(['Open', 'High', 'Low', 'Close', 'Adj Close', 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26'])
    dummy_array = np.zeros((predictions.shape[0], n_features - 1))  # Create dummy array for inverse transform
    predictions_full = np.concatenate([predictions, dummy_array], axis=1)
    predictions_original_scale = scaler.inverse_transform(predictions_full)[:, 3]  # Only take 'Close' column

    # Tweak predictions to fit historical trend
    # Use a scaling factor based on the last known close price
    scaling_factor = last_known_close / predictions_original_scale[0] if predictions_original_scale[0] != 0 else 1
    tweaked_predictions = predictions_original_scale * scaling_factor

    # Introduce more variation to the tweaked predictions
    variation_percentage = 0.03 # 2 variation
    random_variation = np.random.uniform(-variation_percentage, variation_percentage, size=tweaked_predictions.shape)
    tweaked_predictions += tweaked_predictions * random_variation

    # Ensure no negative prices
    tweaked_predictions = np.maximum(tweaked_predictions, 0)

    return tweaked_predictions, forecasted_dates

def prediction_page(data, n_timesteps):
    """Streamlit function to display the stock price prediction page."""
    st.title("Stock Price Prediction")

    # Stock selection
    ticker = st.selectbox("Select Stock:", data['Ticker'].unique())
    
    # Time range selection
    time_range = st.selectbox("Select Prediction Time Range:", ["1 Week", "1 Month", "6 Months"])

    # Load the LSTM model for the selected ticker
    model = load_model(f"lstm_models/{ticker}_lstm_model.h5")

    # Filter stock data for the selected ticker
    stock_data = data[data['Ticker'] == ticker]
    stock_data = stock_data.sort_values('Date')

    # Load scaler and fit it on data
    scaler = MinMaxScaler(feature_range=(0, 1))
    features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26']
    scaler.fit(stock_data[features])

    # Make predictions
    predictions, forecasted_dates = predict_stock(ticker, model, stock_data, time_range, n_timesteps, scaler)

    # Prepare data for plotting
    last_month_data = stock_data[stock_data['Date'] >= (stock_data['Date'].max() - pd.DateOffset(months=3))]
    
    # Create a DataFrame for plotting
    forecasted_df = pd.DataFrame({
        'Date': forecasted_dates,
        'Predicted Close': predictions
    })

    # Combine historical and forecasted data
    combined_df = pd.concat([last_month_data[['Date', 'Close']], forecasted_df], ignore_index=True)

    # Plotting with Plotly
    fig = go.Figure()

    # Add historical data trace
    fig.add_trace(go.Scatter(
        x=combined_df['Date'],
        y=combined_df['Close'],
        mode='lines',
        name='Historical Close',
        line=dict(color='#1f77b4')  # Blue color for historical data
    ))

    # Add predicted data trace
    fig.add_trace(go.Scatter(
        x=forecasted_df['Date'],
        y=forecasted_df['Predicted Close'],
        mode='lines+markers',
        name='Predicted Close',
        line=dict(color='red'),  # Red for predictions
        marker=dict(size=8),
        hoverinfo='text',
        text=[f"Predicted Close: {value:.2f}" for value in predictions]
    ))

    # Update layout
    fig.update_layout(
        title=f"Predictions for {ticker} - Next {time_range}",
        xaxis_title='Date',
        yaxis_title='Stock Price',
        hovermode='x unified',
        template='plotly_white',  # Clean template for better visibility
        margin=dict(l=40, r=40, t=40, b=40),  # Consistent margins
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    st.plotly_chart(fig, use_container_width=True)

    # Optionally, display the predictions in a table
    if st.checkbox("Show Prediction Values"):
        st.dataframe(forecasted_df)

