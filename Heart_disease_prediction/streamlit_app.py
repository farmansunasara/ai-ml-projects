import streamlit as st
import requests
import pandas as pd
import base64

API_URL = "http://127.0.0.1:8000/predict/"

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# ------------ LOAD LOCAL IMAGE AS BASE64 ------------
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_path = "C:/Users/farma/OneDrive/Pictures/Screenshots 1/healthbg.png"
base64_img = get_base64(img_path)

# ------------ DARK GLASSMORPHISM STYLE ------------
page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{base64_img}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
    background: rgba(0, 0, 0, 0);
}}
[data-testid="stToolbar"] {{
    right: 2rem;
}}

div.block-container {{
    background: rgba(0, 0, 0, 0.5);  /* DARK background */
    border-radius: 20px;
    padding: 2.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: #fff;
}}

label, .stTextInput, .stSelectbox, .stNumberInput {{
    color: #fff !important;
}}

.stButton>button {{
    background-color: #ff4b4b;
    color: white;
    border-radius: 12px;
    padding: 0.6rem 1.5rem;
    font-size: 1rem;
    transition: 0.3s;
    border: none;
}}
.stButton>button:hover {{
    background-color: #ff1f1f;
    transform: scale(1.05);
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ------------ TITLE ------------
st.markdown("<h1 style='text-align: center; color: #fff;'>❤️ Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#ccc;'>Fill in the details below to check your heart disease risk.</p>", unsafe_allow_html=True)

# ------------ FORM ------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=1, max_value=120, value=30)
        Sex = st.selectbox("Sex", ["M", "F"])
        ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        RestingBP = st.number_input("Resting BP", min_value=0, max_value=250, value=120)
        Cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200)
        FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

    with col2:
        RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        MaxHR = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
        ExerciseAngina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
        Oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("🔍 Predict")

# ------------ API CALL ------------
if submitted:
    input_data = {
        "Age": int(Age),
        "Sex": Sex,
        "ChestPainType": ChestPainType,
        "RestingBP": int(RestingBP),
        "Cholesterol": int(Cholesterol),
        "FastingBS": int(FastingBS),
        "RestingECG": RestingECG,
        "MaxHR": int(MaxHR),
        "ExerciseAngina": ExerciseAngina,
        "Oldpeak": float(Oldpeak),
        "ST_Slope": ST_Slope
    }

    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            result = response.json()["prediction"]
            if result == 0:
                st.success("✅ Normal — No signs of heart disease detected.")
            else:
                st.error("⚠️ Please consult with a doctor — chances of heart attack detected.")
        else:
            st.warning("⚠️ Something went wrong. Try again later.")
    except Exception as e:
        st.error(f"❌ Error connecting to API: {e}")
