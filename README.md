# 📈 Stock Price Predictor (ML Based)

A **Streamlit-based web app** that predicts stock prices using **Machine Learning models** such as Random Forest, Gradient Boosting, and Linear Regression.  
It allows users to choose a stock ticker, define a time period, visualize closing prices, and compare predictions with real prices.  

---

## 🚀 Features

- 🔎 Select from **popular stocks** (AAPL, GOOGL, AMZN, MSFT, TSLA, META, NFLX, NVDA)  
- 📂 Fetch **historical stock data** using Yahoo Finance API (`yfinance`)  
- 📉 Visualize stock **closing prices**  
- ⚙️ **Preprocess data** with MinMax scaling and sequence generation  
- 🧠 Train & compare **3 machine learning models**:
  - Random Forest Regressor  
  - Gradient Boosting Regressor  
  - Linear Regression  
- 📊 Compare **real prices vs predicted prices** in charts  

---

## 🛠️ Tech Stack

- **Python 3.8+**  
- [Streamlit](https://streamlit.io/) – Web app framework  
- [yFinance](https://pypi.org/project/yfinance/) – Stock market data  
- [NumPy](https://numpy.org/) – Numerical operations  
- [Pandas](https://pandas.pydata.org/) – Data manipulation  
- [Matplotlib](https://matplotlib.org/) – Data visualization  
- [scikit-learn](https://scikit-learn.org/) – ML models & preprocessing  

---

## 📥 Installation

Clone this repository and install dependencies:  

```bash
git clone https://github.com/your-username/stock-price-predictor.git
cd stock-price-predictor
pip install -r requirements.txt
