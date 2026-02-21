import streamlit as st
import pandas as pd
import joblib

# =========================
# ส่วนหัวเว็บ
# =========================
st.set_page_config(
    page_title="ระบบติดตามสภาพอากาศและพยากรณ์อุณหภูมิ",
    layout="centered"
)

st.title("🌦️ ระบบติดตามสภาพอากาศและพยากรณ์อุณหภูมิ")
st.write(
    "เว็บแอปนี้ใช้สำหรับอัปโหลดข้อมูลสภาพอากาศจาก IoT Sensor "
    "และแสดงผลการพยากรณ์อุณหภูมิด้วย Machine Learning"
)

st.divider()

# =========================
# โหลดโมเดล
# =========================
model = joblib.load("temp_model.pkl")

# =========================
# อัปโหลดไฟล์
# =========================
uploaded_file = st.file_uploader(
    "📂 อัปโหลดไฟล์ข้อมูลสภาพอากาศ (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    # =========================
    # โหลดข้อมูล
    # =========================
    df = pd.read_csv(uploaded_file)

    # สร้าง lag feature
    df["temp_lag1"] = df["temp"].shift(1)
    df = df.dropna()

    # =========================
    # เตรียมข้อมูลทำนาย
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
    # พยากรณ์
    # =========================
    df["predicted_temp"] = model.predict(X)

    # =========================
    # แสดงผล
    # =========================
    st.subheader("📈 ผลการพยากรณ์อุณหภูมิ")

    chart_df = df[["temp", "predicted_temp"]]
    st.line_chart(chart_df)

    st.subheader("📋 ตัวอย่างข้อมูลที่อัปโหลด")
    st.dataframe(df.head(20))

    st.success(f"✅ พยากรณ์สำเร็จทั้งหมด {len(df)} รายการ")

else:
    st.info("กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มการพยากรณ์")
