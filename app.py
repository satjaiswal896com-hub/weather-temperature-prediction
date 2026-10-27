
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Weather Temperature Prediction",
    page_icon="🌦️",
    layout="wide"
)

CSV_FILE = "weather.csv"

FEATURES = [
    "month", "day_of_year", "temp_lag_1", "temp_lag_2",
    "temp_lag_3", "humidity_lag_1", "pressure_lag_1",
    "wind_lag_1", "temp_roll_7"
]


def generate_sample_data(days=1500):
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    day = dates.dayofyear.values

    temperature = (
        25 + 10 * np.sin(2 * np.pi * (day - 100) / 365)
        + np.random.normal(0, 1.5, days)
    )
    humidity = 60 - 0.8 * (temperature - 25) + np.random.normal(0, 5, days)
    pressure = 1013 + np.random.normal(0, 4, days)
    wind_speed = np.abs(8 + np.random.normal(0, 3, days))

    return pd.DataFrame({
        "date": dates,
        "temperature": temperature.round(2),
        "humidity": np.clip(humidity, 10, 100).round(2),
        "pressure": pressure.round(2),
        "wind_speed": wind_speed.round(2)
    })


def prepare_data(df):
    df = df.copy()

    required = [
        "date", "temperature", "humidity",
        "pressure", "wind_speed"
    ]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=required)
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    if len(df) < 15:
        raise ValueError("Please provide at least 15 valid daily records.")

    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    for lag in (1, 2, 3):
        df[f"temp_lag_{lag}"] = df["temperature"].shift(lag)

    df["humidity_lag_1"] = df["humidity"].shift(1)
    df["pressure_lag_1"] = df["pressure"].shift(1)
    df["wind_lag_1"] = df["wind_speed"].shift(1)

    df["temp_roll_7"] = (
        df["temperature"].shift(1).rolling(7).mean()
    )

    data = df.dropna(subset=FEATURES).reset_index(drop=True)
    return df, data


st.title("🌦️ Weather Temperature Prediction")
st.write(
    "A machine learning project to forecast next-day average "
    "temperature using historical weather observations."
)

st.sidebar.header("Data Settings")
uploaded_file = st.sidebar.file_uploader(
    "Upload weather.csv",
    type=["csv"]
)

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    data_source = "Uploaded CSV"
elif os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
    raw_df = pd.read_csv(CSV_FILE)
    data_source = "Local weather.csv"
else:
    raw_df = generate_sample_data()
    raw_df.to_csv(CSV_FILE, index=False)
    data_source = "Generated sample data"

st.sidebar.info(f"Data source: {data_source}")

try:
    df, data = prepare_data(raw_df)

    if len(data) < 10:
        st.error("Not enough usable records after preprocessing.")
        st.stop()

    split = int(len(data) * 0.8)
    train = data.iloc[:split]
    test = data.iloc[split:]

    X_train = train[FEATURES]
    y_train = train["temperature"]
    X_test = test[FEATURES]
    y_test = test["temperature"]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
    }

    predictions = {}
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred

        results[name] = {
            "MAE (°C)": mean_absolute_error(y_test, pred),
            "RMSE (°C)": np.sqrt(mean_squared_error(y_test, pred)),
            "R² Score": r2_score(y_test, pred)
        }

    best_model_name = min(
        results,
        key=lambda name: results[name]["MAE (°C)"]
    )

    # Forecast using the most recent available observation
    latest = df.iloc[-1]
    tomorrow = latest["date"] + pd.Timedelta(days=1)

    next_features = pd.DataFrame([{
        "month": tomorrow.month,
        "day_of_year": tomorrow.dayofyear,
        "temp_lag_1": df["temperature"].iloc[-1],
        "temp_lag_2": df["temperature"].iloc[-2],
        "temp_lag_3": df["temperature"].iloc[-3],
        "humidity_lag_1": df["humidity"].iloc[-1],
        "pressure_lag_1": df["pressure"].iloc[-1],
        "wind_lag_1": df["wind_speed"].iloc[-1],
        "temp_roll_7": df["temperature"].tail(7).mean()
    }])[FEATURES]

    forecast = models[best_model_name].predict(next_features)[0]

    st.caption(
        f"Data covers {df['date'].min().date()} to "
        f"{df['date'].max().date()}."
    )

    st.subheader("Tomorrow's Forecast")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Temperature", f"{forecast:.2f} °C")
    c2.metric("Forecast Date", str(tomorrow.date()))
    c3.metric("Best Model", best_model_name)

    st.info(
        "This is an ML-based estimate, not an official weather forecast. "
        "Prediction quality depends on the accuracy and location of your data."
    )

    st.subheader("Model Performance")

    result_df = pd.DataFrame(results).T
    st.dataframe(
        result_df.style.format("{:.3f}"),
        use_container_width=True
    )

    st.subheader("Actual vs Predicted Temperature")

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    dates = test["date"].iloc[-100:]
    actual = y_test.iloc[-100:]

    ax1.plot(dates, actual, label="Actual", linewidth=2)

    for name, pred in predictions.items():
        ax1.plot(
            dates,
            pred[-len(dates):],
            label=name,
            alpha=0.8
        )

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Actual vs Predicted — Test Period")
    ax1.legend()
    ax1.grid(alpha=0.3)
    fig1.autofmt_xdate()
    st.pyplot(fig1)
    plt.close(fig1)

    st.subheader("Feature Importance")

    rf = models["Random Forest"]
    importance = pd.Series(
        rf.feature_importances_,
        index=FEATURES
    ).sort_values()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    importance.plot(kind="barh", ax=ax2)
    ax2.set_xlabel("Importance")
    ax2.set_title("Random Forest Feature Importance")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.subheader("Weather Dataset")
    st.dataframe(df.tail(20), use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Processed Weather Data",
        data=csv_data,
        file_name="processed_weather.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"Could not process the weather data: {e}")
    st.write(
        "Check that your CSV contains date, temperature, humidity, "
        "pressure, and wind_speed columns."
    )
