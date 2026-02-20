import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Weather Monitoring Application")
st.title("🌤 Weather Monitoring Application")

st.write("อัปโหลดไฟล์ข้อมูลอากาศ (unseen data) เพื่อพยากรณ์อุณหภูมิรายชั่วโมง")

# โหลดโมเดล
model = joblib.load("temp_model.pkl")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required_cols = ["humudity", "pressure", "wind", "uv"]
    if all(col in df.columns for col in required_cols):
        X = df[required_cols]

        df["predicted_temp"] = model.predict(X)

        st.subheader("📊 ผลการพยากรณ์อุณหภูมิ")
        st.dataframe(df.head())

        st.subheader("📈 กราฟอุณหภูมิที่พยากรณ์ได้")
        st.line_chart(df["predicted_temp"])

        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download prediction_output.csv",
            csv_out,
            "prediction_output.csv",
            "text/csv"
        )

        st.success("พยากรณ์อุณหภูมิเรียบร้อยแล้ว ✅")
    else:
        st.error("ไฟล์ CSV ต้องมีคอลัมน์: humudity, pressure, wind, uv")