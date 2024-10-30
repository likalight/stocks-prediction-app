import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Set the path for your dataset
DATA_PATH = "./final_dataset.csv"

# Load the dataset
data = pd.read_csv(DATA_PATH)

# Convert Date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Historical Data Page Function
def historical_data_page():
    st.title("Historical Stock Data")
    st.write("Explore historical data for different stocks.")
    
    # Get unique tickers
    tickers = data['Ticker'].unique()
    
    selected_ticker = st.selectbox("Select a Ticker", tickers)
    
    # Filter data by selected ticker
    stock_data = data[data['Ticker'] == selected_ticker]

    # Date range selection
    min_date = stock_data['Date'].min().date()
    max_date = stock_data['Date'].max().date()
    
    start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)
    
    # Filter data by date range
    mask = (stock_data['Date'] >= pd.to_datetime(start_date)) & (stock_data['Date'] <= pd.to_datetime(end_date))
    filtered_stock_data = stock_data.loc[mask]

    st.subheader(f"Historical Data for {selected_ticker} from {start_date} to {end_date}")

    # Plotting Historical Prices with Indicators
    fig = go.Figure()

    # Add closing price
    fig.add_trace(go.Scatter(x=filtered_stock_data['Date'], y=filtered_stock_data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
    
    # Add SMA
    fig.add_trace(go.Scatter(x=filtered_stock_data['Date'], y=filtered_stock_data['SMA_20'], mode='lines', name='SMA 20', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=filtered_stock_data['Date'], y=filtered_stock_data['SMA_50'], mode='lines', name='SMA 50', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=filtered_stock_data['Date'], y=filtered_stock_data['SMA_200'], mode='lines', name='SMA 200', line=dict(color='red', width=2)))

    # Add EMA
    fig.add_trace(go.Scatter(x=filtered_stock_data['Date'], y=filtered_stock_data['EMA_12'], mode='lines', name='EMA 12', line=dict(color='purple', width=2)))
    fig.add_trace(go.Scatter(x=filtered_stock_data['Date'], y=filtered_stock_data['EMA_26'], mode='lines', name='EMA 26', line=dict(color='pink', width=2)))

    # Update layout
    fig.update_layout(title=f'Historical Prices and Indicators for {selected_ticker}',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=True,
                      template='plotly_white')

    st.plotly_chart(fig)

    # RSI
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=filtered_stock_data['Date'], y=filtered_stock_data['RSI'], mode='lines', name='RSI', line=dict(color='cyan')))
    fig_rsi.update_layout(title='Relative Strength Index (RSI)',
                          xaxis_title='Date',
                          yaxis_title='RSI',
                          xaxis_rangeslider_visible=True,
                          template='plotly_white')
    st.plotly_chart(fig_rsi)

    # Daily Returns
    fig_returns = go.Figure()
    fig_returns.add_trace(go.Scatter(x=filtered_stock_data['Date'], y=filtered_stock_data['Daily_Return'], mode='lines', name='Daily Return', line=dict(color='magenta')))
    fig_returns.update_layout(title='Daily Returns',
                              xaxis_title='Date',
                              yaxis_title='Return',
                              xaxis_rangeslider_visible=True,
                              template='plotly_white')
    st.plotly_chart(fig_returns)

    # Volatility
    fig_volatility = go.Figure()
    fig_volatility.add_trace(go.Scatter(x=filtered_stock_data['Date'], y=filtered_stock_data['Volatility'], mode='lines', name='Volatility', line=dict(color='brown')))
    fig_volatility.update_layout(title='Volatility',
                                  xaxis_title='Date',
                                  yaxis_title='Volatility',
                                  xaxis_rangeslider_visible=True,
                                  template='plotly_white')
    st.plotly_chart(fig_volatility)
