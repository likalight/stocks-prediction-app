import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor

MODEL_PATH = "lstm_models/"

# Function to tweak predictions (based on your existing tweak method)
def tweak_predictions(predictions_original_scale, last_known_close):
    scaling_factor = last_known_close / predictions_original_scale[0] if predictions_original_scale[0] != 0 else 1
    tweaked_predictions = predictions_original_scale * scaling_factor

    variation_percentage = 0.03
    random_variation = np.random.uniform(-variation_percentage, variation_percentage, size=tweaked_predictions.shape)
    tweaked_predictions += tweaked_predictions * random_variation

    tweaked_predictions = np.maximum(tweaked_predictions, 0)

    return tweaked_predictions

# Prepare input data for LSTM prediction
def prepare_input_data(stock_data, n_timesteps, scaler):
    input_data = stock_data.iloc[-n_timesteps:][['Open', 'High', 'Low', 'Close', 'Adj Close',
                                                 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26']].values
    input_data_scaled = scaler.transform(input_data)
    input_data_scaled = input_data_scaled.reshape((1, n_timesteps, len(input_data[0])))
    return input_data_scaled

# Predict for 7 days with tweaked predictions
def predict_for_7_days(ticker, model, stock_data, n_timesteps, scaler):
    input_data_scaled = prepare_input_data(stock_data, n_timesteps, scaler)

    days_to_predict = 7  # Predict for 7 days
    predictions = []
    last_known_date = stock_data['Date'].max()
    forecasted_dates = []

    for i in range(days_to_predict):
        pred_value_scaled = model.predict(input_data_scaled)[0][0]
        predictions.append(pred_value_scaled)

        new_row = input_data_scaled[:, -1, :].copy()
        new_row[0, 3] = pred_value_scaled  # Update the 'Close' value (index 3)

        input_data_scaled = np.concatenate([input_data_scaled[:, 1:, :], new_row.reshape(1, 1, input_data_scaled.shape[2])], axis=1)
        forecasted_dates.append(last_known_date + timedelta(days=i + 1))

    predictions = np.array(predictions).reshape(-1, 1)

    n_features = len(['Open', 'High', 'Low', 'Close', 'Adj Close', 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26'])
    dummy_array = np.zeros((predictions.shape[0], n_features - 1))
    predictions_full = np.concatenate([predictions, dummy_array], axis=1)
    predictions_original_scale = scaler.inverse_transform(predictions_full)[:, 3]

    return predictions_original_scale, forecasted_dates

# Function to rank stocks based on 7-day prediction
def rank_stock(ticker, model, stock_data, n_timesteps, scaler):
    try:
        predictions_original_scale, forecasted_dates = predict_for_7_days(ticker, model, stock_data, n_timesteps, scaler)

        last_known_close = stock_data['Close'].iloc[-1]
        tweaked_predictions = tweak_predictions(predictions_original_scale, last_known_close)

        final_price = tweaked_predictions[-1]
        return_percentage = ((final_price - last_known_close) / last_known_close) * 100

        return {
            'Ticker': ticker,
            'Predicted Price (7 days)': final_price,
            'Return (%)': return_percentage
        }
    except Exception as e:
        st.error(f"Error with ticker {ticker}: {e}")
        return None

# Run stock ranking in parallel
def rank_stocks_parallel(data, n_timesteps, model_path, scaler):
    ranking_data = []
    tickers = data['Ticker'].unique()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(rank_stock, ticker, load_model(f"{model_path}/{ticker}_lstm_model.h5"),
                            data[data['Ticker'] == ticker], n_timesteps, scaler)
            for ticker in tickers
        ]
        for future in futures:
            result = future.result()
            if result:
                ranking_data.append(result)

    ranking_df = pd.DataFrame(ranking_data)
    ranking_df.sort_values(by='Return (%)', ascending=False, inplace=True)
    return ranking_df

# Custom function to display the ranking DataFrame
def display_ranking_table(ranking_df):
    ranking_df = ranking_df.reset_index(drop=True)

    # Add a "Symbol" column before displaying the table
    ranking_df['Symbol'] = ranking_df['Return (%)'].apply(lambda x: '⬆️' if x > 0 else '⬇️')

    # Define a function to color the return percentages
    def color_return(val):
        color = 'green' if val > 0 else 'red'
        return f'color: {color}'

    st.markdown("""
        <style>
        .dataframe {
            margin: auto;
        }
        </style>
        """, unsafe_allow_html=True)

    # Apply the color formatting to the 'Return (%)' column
    styled_df = (ranking_df[['Ticker', 'Predicted Price (7 days)', 'Return (%)', 'Symbol']]
                .style
                .set_properties(**{'text-align': 'center'})
                .format({'Return (%)': "{:.2f}%"})  # Format the percentage
              .applymap(color_return, subset=['Return (%)']))  # Apply color based on value

    # Adjust the height based on the number of rows (e.g., 40px per row)
    num_rows = len(ranking_df)
    row_height = 40  # Height per row
    table_height = min(740, row_height * num_rows)  # Set a max height of 800

    # Display the styled DataFrame without scrolling, with dynamic height
    st.dataframe(styled_df, use_container_width=True, height=table_height)

# Streamlit function to display ranking page
def ranking_page(data, n_timesteps):
    st.title("Stock Ranking Based on 7-Day Prediction")

    scaler = MinMaxScaler(feature_range=(0, 1))
    features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26']
    scaler.fit(data[features])

    ranking_df = rank_stocks_parallel(data, n_timesteps, MODEL_PATH, scaler)

    # Ensure the Symbol column is created before accessing it
    ranking_df['Symbol'] = ranking_df['Return (%)'].apply(lambda x: '⬆️' if x > 0 else '⬇️')

    best_stock = ranking_df.iloc[0]

    # Display the Top Performing Stock centered
    st.markdown(f"""
        <div style="text-align: center;">
            <strong>Top Performing Stock: {best_stock['Ticker']} with a predicted return of {best_stock['Return (%)']:.2f}% {best_stock['Symbol']}</strong>
        </div>
    """, unsafe_allow_html=True)

    display_ranking_table(ranking_df)
