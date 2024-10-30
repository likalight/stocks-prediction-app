import streamlit as st
import pandas as pd
from historical_data import historical_data_page
from ranking import ranking_page
from prediction import prediction_page
from datetime import timedelta

# Load the stock dataset
data = pd.read_csv("./final_dataset.csv")
data['Date'] = pd.to_datetime(data['Date'])

def main():
    st.set_page_config(page_title="Stock Prediction App", layout="wide")

    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Go to", ["Main Page", "Historical Data", "Ranking Page", "Prediction Page"])  # Added Prediction Page option

    if options == "Main Page":
        st.title("📈 ZipFin Stock Prediction App 📈")
        st.caption("Created By: Su Myat (2403062), Kalai (2403281), Nayli (2403067), Xin Rong (2403192), Zharfan (2402959)")
        st.write("Welcome to our Stock Prediction Application!")
        st.write("This application will allow you to explore historical stock data for 20 different US stocks from 01/01/2013 to 31/08/2024.")
        st.write("Using the historical data, we built an LSTM model for users to view predictions for the next 1 week, 1 month and 6 month.")
        st.write("At a glance, you will also be able to view the historical performance of stocks and key technical indicators.")

    elif options == "Historical Data":
        historical_data_page()

    elif options == "Ranking Page":
        ranking_page(data, n_timesteps=60)  # Call the ranking page function

    elif options == "Prediction Page":
        prediction_page(data, n_timesteps=60)  # Call the prediction page function

if __name__ == "__main__":
    main()