import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# --- Streamlit Page ---
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor (ML Based)")

# --- Sidebar Options ---
st.sidebar.header("Stock Settings")
popular_stocks = {
    "Apple (AAPL)": "AAPL",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
    "NVIDIA (NVDA)": "NVDA"
}
ticker_label = st.sidebar.selectbox("Select Stock", options=list(popular_stocks.keys()))
ticker = popular_stocks[ticker_label]

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2022-01-01"))

# --- Predict Button ---
if st.sidebar.button("Predict"):

    st.subheader(f"📊 Loading Data for: {ticker}")
    data = yf.download(ticker, start=start_date, end=end_date)
    st.write(data.tail())

    # --- Plot Raw Closing Price ---
    st.subheader("📉 Closing Price Chart")
    fig, ax = plt.subplots()
    data['Close'].plot(ax=ax, label='Close Price')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{ticker} Closing Price")
    st.pyplot(fig)

    # --- Preprocessing ---
    st.subheader("⚙️ Data Preprocessing...")
    close_data = data[['Close']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_data)

    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i].flatten())
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y).ravel()

    # Train/test split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # --- Models ---
    st.subheader("🧠 Training Models...")
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression()
    }

    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        predictions[name] = scaler.inverse_transform(preds.reshape(-1, 1))

    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # --- Plotting All Predictions ---
    st.subheader("🔍 Model Comparison - Predicted vs Real Prices")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(real_prices, label="Real Price", linewidth=2)
    for name, preds in predictions.items():
        ax2.plot(preds, label=name)
    ax2.set_title("Model Predictions vs Real Price")
    ax2.legend()
    st.pyplot(fig2)
 