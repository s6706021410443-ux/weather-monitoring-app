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
# Theme Toggle
# =========================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

col1, col2 = st.columns([6,1])
with col2:
    theme_choice = st.radio(
        "Theme",
        ["🌙 Dark", "🌞 Light"],
        index=0 if st.session_state.theme == "dark" else 1,
        label_visibility="collapsed"
    )

st.session_state.theme = "dark" if "Dark" in theme_choice else "light"

def apply_theme(theme):
    if theme == "dark":
        st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        section[data-testid="stSidebar"] {
            background-color: #161b22;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background-color: #f5f7fa;
            color: #000000;
        }
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
        }
        </style>
        """, unsafe_allow_html=True)

apply_theme(st.session_state.theme)

# =========================
# Header
# =========================
st.title("🌦️ Weather Monitoring & Temperature Forecast")
st.markdown("""
Upload unseen weather data collected from IoT sensors  
and compare temperature predictions from multiple models.
""")

st.divider()

# =========================
# Load Models (duplicate for demo)
# =========================
@st.cache_resource
def load_models():
    model = joblib.load("temp_model.pkl")
    return {
        "Random Forest (Model A)": model,
        "Random Forest (Model B)": model
    }

models = load_models()

# =========================
# Model Selection
# =========================
st.subheader("🧠 Model Selection")
selected_models = st.multiselect(
    "Select model(s) for prediction",
    options=list(models.keys()),
    default=["Random Forest (Model A)"]
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

    st.subheader("📋 Prediction Preview")
    st.dataframe(result_df.head(20), use_container_width=True)

    st.success(f"✅ Prediction completed for {len(result_df)} records")

else:
    st.info("⬆️ Please upload a CSV file and select at least one model.")
