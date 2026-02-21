import streamlit as st
import pandas as pd
import joblib

st.title("Weather Monitoring & Temperature Forecast")

model = joblib.load("temp_model.pkl")

uploaded_file = st.file_uploader("Upload CSV file")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = ['temp','humudity','pressure','wind','rain drop','uv']
    if not all(col in df.columns for col in required_cols):
        st.error("CSV file missing required columns")
        st.stop()

    df['temp_lag1'] = df['temp'].shift(1)
    df = df.dropna()

    X = df[['temp_lag1','humudity','pressure','wind','rain drop','uv']]
    df['predicted_temp'] = model.predict(X)

    st.subheader("Temperature Prediction Result")
    st.line_chart(df[['temp','predicted_temp']])

    st.subheader("Uploaded Data")
    st.dataframe(df.head(20))

    st.success(f"Prediction completed for {len(df)} records")
