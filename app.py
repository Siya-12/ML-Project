import streamlit as st

import traceback

try:
    
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Insurance Predictor", page_icon="💰", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #2c3e50;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
        background-color: #4CAF50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load model safely
model = joblib.load("model.joblib")

# Title
st.markdown('<div class="title">💰 Medical Insurance Cost Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict your insurance charges instantly</div>', unsafe_allow_html=True)

# User Inputs (in columns)
col1, col2 = st.columns(2)

with col1:
    age = st.slider("🎂 Age", 18, 100, 25)
    bmi = st.number_input("⚖️ BMI", 10.0, 50.0, 25.0)

with col2:
    children = st.slider("👶 Children", 0, 5, 0)
    sex = st.selectbox("🧑 Gender", ["male", "female"])

smoker = st.selectbox("🚬 Smoker", ["yes", "no"])
region = st.selectbox("📍 Region", ["southwest", "southeast", "northwest", "northeast"])

# Encoding (same as training)
sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0

region_map = {
    "southwest": 0,
    "southeast": 1,
    "northwest": 2,
    "northeast": 3
}
region_encoded = region_map[region]

# Predict
if st.button("🔍 Predict Insurance Cost"):
    features = np.array([[age, sex, bmi, children, smoker, region_encoded]])
    prediction = model.predict(features)

    st.markdown("### 💡 Estimated Cost")
    st.success(f"₹ {prediction[0]:,.2f}")

    # Extra insight
    if smoker:
        st.warning("⚠️ Smoking significantly increases insurance cost!")
    if bmi > 30:
        st.info("💡 High BMI may increase premium.")

    pass
except Exception as e:
    import streamlit as st
    st.error(str(e))
    st.text(traceback.format_exc())

