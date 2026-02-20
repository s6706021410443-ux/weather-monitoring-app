import streamlit as st
import pandas as pd

st.set_page_config(page_title="Weather Monitoring Application", layout="centered")

st.title("🌦 Weather Monitoring Application")
st.write(
    "อัปโหลดไฟล์ข้อมูล **unseen data (1 สัปดาห์สุดท้าย)** "
    "เพื่อแสดงผลการพยากรณ์อุณหภูมิรายชั่วโมง"
)

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ==== ตรวจสอบคอลัมน์ตามโจทย์ ====
    required_cols = ["datetime", "actual_temp", "predicted_temp"]

    if not all(col in df.columns for col in required_cols):
        st.error(
            "ไฟล์ CSV ต้องมีคอลัมน์: datetime, actual_temp, predicted_temp"
        )
    else:
        # ==== Data Preparation (เบื้องต้นใน Deployment) ====
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.dropna()
        df = df.drop_duplicates()

        st.success("โหลดข้อมูลสำเร็จ")

        # ==== แสดงตาราง ====
        st.subheader("📄 ข้อมูลผลการพยากรณ์")
        st.dataframe(df)

        # ==== Visualization ====
        st.subheader("📈 Actual vs Predicted Temperature")

        chart_df = df.set_index("datetime")[["actual_temp", "predicted_temp"]]
        st.line_chart(chart_df)

        # ==== Download ====
        st.download_button(
            label="📥 Download CSV",
            data=df.to_csv(index=False),
            file_name="weather_prediction_result.csv",
            mime="text/csv"
        )