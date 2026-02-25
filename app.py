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
    เว็บแอปนี้ใช้สำหรับ **อัปโหลดข้อมูลสภาพอากาศจาก IoT Sensors**
    และทำการ **พยากรณ์อุณหภูมิรายชั่วโมง (Hourly Temperature Forecast)**

    ข้อมูลที่อัปโหลดจะถูกมองว่าเป็น **unseen data**
    เพื่อแสดงการทำงานของระบบในขั้นตอน Deployment
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
    "อัปโหลดไฟล์ CSV ที่ได้จาก IoT Sensor",
    type=["csv"]
)

if uploaded_file is not None:

    # =========================
    # Load Data
    # =========================
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Preparation")

    # แปลงเวลาเป็น datetime
    df["datetime"] = pd.to_datetime(df["datetime"])

    # เรียงตามเวลา
    df = df.sort_values("datetime")

    # สร้าง lag feature
    df["temp_lag1"] = df["temp"].shift(1)

    # ลบแถวที่มีค่า NaN จาก lag
    df = df.dropna()

    # ใช้เวลาเป็น index
    df = df.set_index("datetime")

    st.success("✅ Data prepared successfully")

    # =========================
    # Feature Selection
    # =========================
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
    df["Predicted Temperature (°C)"] = model.predict(X)

    # =========================
    # Visualization
    # =========================
    st.divider()
    st.subheader("📈 Hourly Temperature Prediction")

    result_df = df[
        ["temp", "Predicted Temperature (°C)"]
    ].rename(
        columns={"temp": "Actual Temperature (°C)"}
    )

    st.line_chart(result_df)

    # =========================
    # Data Preview
    # =========================
    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(
        result_df.reset_index().head(20),
        use_container_width=True
    )

    # =========================
    # Status
    # =========================
    st.success(f"✅ Prediction completed for {len(result_df)} records")

else:
    st.info("⬆️ กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มการพยากรณ์อุณหภูมิ")
