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
    Upload **unseen weather data** collected from IoT sensors  
    and compare **temperature predictions** from different machine learning models.
    """
)

st.divider()

# =========================
# Load Models
# =========================
@st.cache_resource
def load_models():
    return {
        "Random Forest": joblib.load("temp_model.pkl"),
        "Linear Regression": joblib.load("temp_model_lr.pkl")
    }

models = load_models()

# =========================
# Model Selection
# =========================
st.subheader("🧠 Model Selection")
selected_models = st.multiselect(
    "Select model(s) for prediction",
    options=list(models.keys()),
    default=["Random Forest"]
)

st.divider()

# =========================
# File Upload
# =========================
uploaded_file = st.file_uploader(
    "📂 Upload Weather CSV file",
    type=["csv"]
)

if uploaded_file and selected_models:

    # =========================
    # Data Preparation
    # =========================
    df = pd.read_csv(uploaded_file)

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
    result_df = pd.DataFrame()
    result_df["Actual Temperature (°C)"] = df["temp"]

    for model_name in selected_models:
        result_df[f"{model_name} Prediction (°C)"] = models[model_name].predict(X)

    # =========================
    # Visualization
    # =========================
    st.subheader("📈 Temperature Prediction Comparison")
    st.line_chart(result_df)

    # =========================
    # Data Preview
    # =========================
    st.subheader("📋 Prediction Result Preview")
    st.dataframe(result_df.head(20), use_container_width=True)

    st.success(
        f"✅ Prediction completed using {len(selected_models)} model(s) "
        f"for {len(result_df)} records"
    )

else:
    st.info("⬆️ Please upload a CSV file and select at least one model.")
