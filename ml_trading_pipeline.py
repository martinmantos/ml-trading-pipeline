# MACHINE LEARNING + PATTERN RECOGNITION TRADING SYSTEM (FOREX & STOCKS)
# === Imports ===
import numpy as np
import pandas as pd
import yfinance as yf
import talib
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import datetime
import pickle
import streamlit as st
import sqlite3

# === Step 1: Data Collection ===
def download_data(symbol="EURUSD=X", start="2020-01-01", end="2024-01-01"):
    data = yf.download(symbol, start=start, end=end)
    data.dropna(inplace=True)
    return data

# === Step 2: Feature Engineering ===
def generate_indicators(df):
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], _, _ = talib.MACD(df['Close'])
    df['SMA50'] = talib.SMA(df['Close'], timeperiod=50)
    df['SMA200'] = talib.SMA(df['Close'], timeperiod=200)
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df.dropna(inplace=True)
    return df

# === Step 3: Pattern Recognition ===
def extract_patterns(df, window=20):
    patterns = []
    for i in range(len(df) - window):
        segment = df['Close'].iloc[i:i+window].values
        normalized = (segment - segment[0]) / segment[0]
        patterns.append(normalized)
    return np.array(patterns)

# === Step 4: Pattern Clustering ===
def cluster_patterns(patterns, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(patterns)
    return kmeans

# === Step 4.5: Store Patterns in SQLite DB ===
def store_patterns_in_db(patterns, labels):
    conn = sqlite3.connect('patterns.db')
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern TEXT,
                        label INTEGER
                    )""")
    for p, l in zip(patterns, labels):
        cursor.execute("INSERT INTO patterns (pattern, label) VALUES (?, ?)", (','.join(map(str, p)), int(l)))
    conn.commit()
    conn.close()

# === Step 5: Label Creation for Supervised Learning ===
def create_labels(df, future_window=5):
    df['Future_Close'] = df['Close'].shift(-future_window)
    df['Target'] = (df['Future_Close'] > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df

# === Step 6: ML Model for Prediction ===
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# === Step 7: LSTM Deep Pattern Recognition ===
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === Step 8: Backtesting Engine ===
def backtest_strategy(df, model, features):
    df['Signal'] = model.predict(features)
    df['Returns'] = df['Close'].pct_change()
    df['Strategy'] = df['Signal'].shift(1) * df['Returns']
    df['Cumulative_Market'] = df['Returns'].cumsum()
    df['Cumulative_Strategy'] = df['Strategy'].cumsum()
    return df

# === Step 9: Streamlit Dashboard ===
def run_dashboard():
    st.title("ML Pattern Recognition Trading Dashboard")

    st.subheader("Batch Test to Find Optimal Pair for $100 Account")

    candidate_symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X", "USDCHF=X", "XAUUSD=X"]
    start = st.date_input("Start Date", value=datetime.date(2020, 1, 1))
    end = st.date_input("End Date", value=datetime.date(2024, 1, 1))

    if st.button("Find Best Pair"):
        performance_data = []

        for symbol in candidate_symbols:
            try:
                df = download_data(symbol, str(start), str(end))
                df = generate_indicators(df)
                df = create_labels(df)
                features = df[['RSI', 'MACD', 'SMA50', 'SMA200', 'ATR']].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(features)
                y = df['Target'].values

                model = train_model(X_scaled, y)
                df_result = backtest_strategy(df.copy(), model, X_scaled)

                final_return = df_result['Cumulative_Strategy'].iloc[-1]
                win_rate = (df_result['Signal'] == df_result['Target']).mean()
                performance_data.append((symbol, round(final_return, 4), round(win_rate * 100, 2)))
            except Exception as e:
                performance_data.append((symbol, 'error', 'error'))

        result_df = pd.DataFrame(performance_data, columns=['Symbol', 'Strategy Return', 'Prediction Accuracy (%)'])
        st.dataframe(result_df.sort_values(by='Strategy Return', ascending=False).reset_index(drop=True))

# === Main Execution ===
if __name__ == "__main__":
    run_dashboard()
