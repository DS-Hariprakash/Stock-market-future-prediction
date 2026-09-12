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

# --- Sidebar Options (kept minimal: ticker + one action button) ---
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

# Fixed lookback window instead of manual date pickers.
LOOKBACK_YEARS = 5
end_date = pd.Timestamp.today().normalize()
start_date = end_date - pd.DateOffset(years=LOOKBACK_YEARS)

# --- Load Data & Predict Button ---
if st.sidebar.button("Load Data & Predict"):

    st.subheader(f"📊 Loading Data for: {ticker} (last {LOOKBACK_YEARS} years)")
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.error(f"No data returned for {ticker}. Try again later.")
        st.stop()

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

    # Split into train/test on raw prices first, then fit the scaler on
    # training data only, to avoid leaking test-set price range into training.
    sequence_length = 60
    split_idx = int(0.8 * len(close_data))
    train_raw = close_data.iloc[:split_idx]
    test_raw = close_data.iloc[split_idx - sequence_length:]  # keep lookback context

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_raw)
    scaled_test = scaler.transform(test_raw)

    def make_sequences(scaled_series):
        X, y = [], []
        for i in range(sequence_length, len(scaled_series)):
            X.append(scaled_series[i - sequence_length:i].flatten())
            y.append(scaled_series[i])
        return np.array(X), np.array(y).ravel()

    X_train, y_train = make_sequences(scaled_train)
    X_test, y_test = make_sequences(scaled_test)

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
 