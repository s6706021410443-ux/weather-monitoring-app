import streamlit as st
import pandas as pd
import joblib

st.title("Weather Monitoring & Temperature Forecast")

model = joblib.load("temp_model.pkl")

uploaded_file = st.file_uploader("Upload CSV file")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['temp_lag1'] = df['temp'].shift(1)
    df = df.dropna()

    X = df[['temp_lag1','humudity','pressure','wind','rain drop','uv']]
    df['predicted_temp'] = model.predict(X)

    st.line_chart(df[['temp','predicted_temp']])