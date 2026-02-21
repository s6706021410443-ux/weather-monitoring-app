import streamlit as st
import pandas as pd
import joblib

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Weather Monitoring & Temperature Forecast",
    page_icon="🌦️",
    layout="wide"
)

# =========================
# Header
# =========================
st.title("🌦️ Weather Monitoring & Temperature Forecast")
st.markdown(
    """
    This web application is designed to **upload weather data collected from IoT sensors**
    and **predict hourly temperature** using a Machine Learning model.

    The uploaded file is treated as **unseen data** (not used in model training),
    which demonstrates real-world deployment of the forecasting system.
    """
)

st.divider()

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return joblib.load("temp_model.pkl")

model = load_model()

# =========================
# File Upload
# =========================
st.subheader("📂 Upload Weather Data (CSV)")
uploaded_file = st.file_uploader(
    "Please upload a CSV file containing weather sensor data",
    type=["csv"]
)

if uploaded_file is not None:

    # =========================
    # Load & Prepare Data
    # =========================
    df = pd.read_csv(uploaded_file)

    st.markdown("### 🔍 Data Preparation")
    st.write("Creating lag feature and preparing data for prediction.")

    df["temp_lag1"] = df["temp"].shift(1)
    df = df.dropna()

    features = [
        "temp_lag1",
        "humudity",
        "pressure",
        "wind",
        "rain drop",
        "uv"
    ]

    X = df[features]

    # =========================
    # Prediction
    # =========================
    df["predicted_temp"] = model.predict(X)

    # =========================
    # Visualization
    # =========================
    st.divider()
    st.subheader("📈 Temperature Prediction Result")

    chart_df = df[["temp", "predicted_temp"]].rename(
        columns={
            "temp": "Actual Temperature (°C)",
            "predicted_temp": "Predicted Temperature (°C)"
        }
    )

    st.line_chart(chart_df)

    # =========================
    # Data Preview
    # =========================
    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # =========================
    # Status
    # =========================
    st.success(f"✅ Prediction completed for {len(df)} records")

else:
    st.info("⬆️ Please upload a CSV file to start temperature prediction.")
