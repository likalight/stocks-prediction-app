# INF1002 - Programming Fundamentals Python Project

### Stock Price Prediction Application
Authored by : Su Myat (2403062), Kalai (2403281), Nayli (2403067), Xin Rong (2403192), Zharfan (2402959)
LAB-P14 : Group 3

Our project is a stock price prediction application that uses Long Short-Term Memory (LSTM), for predicting stock prices of various companies. Data collection, transformation, analysis and machine learning model building were all done in the jupyter notebook found in the dataandmodelling folder. We trained 3 different models: Linear Regression, SARIMAX and LSTM and compared the MSE and RMSE to determine the best model for our data. The application is built using Python and Streamlit, which aims to allow users to :
1) explore historical stock data 
2) ranks the stocks based on their predicted returns over a 6-month horizon
3) generate future price predictions for different time horizons (1 week, 1 month, and 6 months)

### Folder Structure
In the ./dataandmodelling folder, you will be able to find the Jupyter notebook we used for data analysis and model training. The output CSVs and exported models are also located in this folder.

The rest of the .py files in the ./P14_Group3_SourceCode folder are required for the application to run. 

### Dataset Used
Our application uses the final_dataset.csv which contains the following fields for each stock:
1) Open: The price at the market open.
2) High: The highest price of the day.
3) Low: The lowest price of the day.
4) Close: The price at the market close.
5) Adj Close: Adjusted closing price (accounts for corporate actions like dividends and splits).
6) Volume: The number of shares traded.
7) SMA_20, SMA_50, SMA_200: Simple moving averages over different time periods.
8) EMA_12, EMA_26: Exponential moving averages over 12 and 26 periods.
9) RSI: Relative strength index.
10) Daily Return, Volatility: Other calculated metrics.


## Running The Application

### Step 1 - Change directory to ensure you are in our project folder
Change directory to the path you have our project folder installed into

```
cd \P14_Group3_PythonProjectSubmission\P14_Group3_SourceCode
```

### Step 2 - Setting up a Virtual Environment

<p>To set up a virtual environment, run the following code in terminal</p>

```
pip install virtualenv
python -m venv env 
```

<p>To activate your virtual environment on WINDOWS , run</p>

```
env\Scripts\activate
```

<p>To activate your virtual environment on MacOS/Linux , run</p>

```
source env/bin/activate
```

### Step 3 - Installing Project Dependencies from requirements.txt

Requirements.txt includes all of the packages we will be using

```
pip install -r requirements.txt
```

### Step 4 - Running our Application

```
streamlit run app.py
```

### Step 5 - Stopping our Application
Go to the terminal where it is running and press Ctrl + C

### Step 6 - Deactivate Virtual Environment
<p>To deactivate your virtual environment after everything, run</p>

```
deactivate
```