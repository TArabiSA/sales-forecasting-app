import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import requests
from streamlit_lottie import st_lottie

from model.simple_lstm import SimpleLSTM, forecast_multistep


# -------------------------------
# Load Lottie animation
# -------------------------------
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_o6spyjnc.json")


# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="lstm model",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

st.title("📊 Sales Forecasting")
st.write("Upload your sales data (`date,sales`) to forecast future sales using an LSTM (NumPy implementation).")


# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload Sales CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["date"])

    # -------------------------------
    # Data Preview + Lottie
    # -------------------------------
    with st.container():
        left_column, right_column = st.columns(2)
        with left_column:
            st.subheader("📄 Sales Data Preview")
            st.write(df.head())
        with right_column:
            st_lottie(lottie_coding, height=250, key="coding")

    # -------------------------------
    # Daily Sales Line Chart
    # -------------------------------
    st.title("Plot Data :chart_with_upwards_trend:")
    st.subheader("📈 Daily Sales Chart")

    fig1 = plt.figure(figsize=(12,6))
    plt.plot(df["date"], df["sales"], label="Daily Sales", color="blue")
    plt.title('Daily Sales Price History')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sale Price PESO (₱)', fontsize=12)
    plt.legend()
    st.pyplot(fig1)

    # -------------------------------
    # Weekly Aggregation + Moving Average
    # -------------------------------
    st.subheader("📊 Weekly Aggregated Sales with Trend")

    df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly_sales = df.groupby("week")["sales"].sum().reset_index()
    weekly_sales["moving_avg"] = weekly_sales["sales"].rolling(window=4).mean()

    fig2, ax = plt.subplots(figsize=(12,6))
    ax.bar(weekly_sales["week"], weekly_sales["sales"], color="skyblue", label="Weekly Sales")
    ax.plot(weekly_sales["week"], weekly_sales["moving_avg"], color="red", linewidth=2, label="4-Week Moving Avg")
    ax.set_title("Weekly Sales Trend")
    ax.set_xlabel("Week")
    ax.set_ylabel("Sales (₱)")
    ax.legend()
    st.pyplot(fig2)

    # -------------------------------
    # Preprocess for LSTM
    # -------------------------------
    scaler = MinMaxScaler(feature_range=(0,1))
    sales = df["sales"].values.reshape(-1,1)
    scaled_sales = scaler.fit_transform(sales)

    SEQ_LEN = 30
    def create_sequences(data, seq_len=SEQ_LEN):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_sales)
    train_size = int(len(X)*0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # -------------------------------
    # Train LSTM
    # -------------------------------
    st.subheader("🧠 Training LSTM Model")
    model = SimpleLSTM(input_size=1, hidden_size=20, output_size=1, lr=0.01)

    epochs = 10
    for epoch in range(epochs):
        loss = 0
        for i in range(len(X_train)):
            y_pred = model.forward(X_train[i])
            loss += np.mean((y_pred - y_train[i])**2)
        loss /= len(X_train)
        st.write(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.6f}")

    # -------------------------------
    # Predictions
    # -------------------------------
    predictions = []
    for i in range(len(X_test)):
        pred = model.forward(X_test[i])
        predictions.append(pred)

    predictions = np.array(predictions)
    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test)

    st.subheader("📉 Predictions vs Actual")
    st.line_chart(pd.DataFrame({
        "Actual": y_test_inv.flatten(),
        "Predicted": predictions_inv.flatten()
    }))

    # -------------------------------
    # Multi-step Forecasting
    # -------------------------------
    st.subheader("🔮 Future Forecasting")

    future_steps = st.slider("Forecast steps", 7, 90, 30)
    seed_sequence = X_test[-1]
    forecast_scaled = forecast_multistep(model, seed_sequence, future_steps)
    forecast_inv = scaler.inverse_transform(forecast_scaled)

    future_dates = pd.date_range(start=df["date"].iloc[-1], periods=future_steps+1, freq="D")[1:]
    forecast_df = pd.DataFrame({"date": future_dates, "forecasted_sales": forecast_inv.flatten()})

    st.write(forecast_df.head(10))

    # Plot forecast
    fig3, ax = plt.subplots(figsize=(10,5))
    ax.plot(df["date"], df["sales"], label="History")
    ax.plot(future_dates, forecast_inv.flatten(), label="Forecast", color="red")
    ax.legend()
    st.pyplot(fig3)
