import streamlit as st
import requests
import base64
import time

# =================== CONFIG ===================
API_URL = "http://127.0.0.1:8000/predict/"
BACKGROUND_IMAGE = "Background.png"

st.set_page_config(page_title="🏦 Loan Approval Predictor", layout="centered")

# =================== BACKGROUND FUNCTION ===================
def set_background(image_path):
    with open(image_path, "rb") as f:
        img_data = f.read()
    b64_img = base64.b64encode(img_data).decode()

    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url("data:image/png;base64,{b64_img}") no-repeat center center fixed;
        background-size: cover;
    }}
    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: transparent !important;
    }}
    .glass {{
        background: rgba(0, 0, 0, 0.6); 
        border-radius: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2.5rem;
    }}
    .title-glow {{
        color: #ffffff;
        text-shadow: 0 0 20px rgba(255,255,255,0.7);
        text-align: center;
    }}
    .subtext {{
        text-align: center;
        color: #cccccc;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }}
    div.stButton > button:first-child {{
        background: linear-gradient(90deg, #4CAF50, #2ecc71);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px rgba(72, 239, 128, 0.5);
    }}
    div.stButton > button:first-child:hover {{
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(72, 239, 128, 0.8);
    }}
    .result-card {{
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# =================== HEADER ===================
st.markdown("<h1 class='title-glow'>🏦 Loan Approval Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Enter applicant details below to check loan approval chances.</p>", unsafe_allow_html=True)
set_background(BACKGROUND_IMAGE)

# =================== FORM ===================
with st.container():
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        with col1:
            Gender = st.selectbox("Gender", ["Male", "Female"])
            Married = st.selectbox("Married", ["Yes", "No"])
            Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
            Age = st.number_input("Age (in years)", min_value=18, max_value=80, value=30)
        with col2:
            ApplicantIncome = st.number_input("Applicant Income", min_value=0.0, step=100.0)
            SideIncome = st.number_input("Side Income (if any)", min_value=0.0, step=100.0)
            CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0, step=100.0)
            LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0.0, step=10.0)
            Loan_Amount_Term = st.number_input("Loan Amount Term (months)", min_value=12.0, max_value=480.0, step=12.0)
            Credit_History = st.selectbox("Credit History", [1, 0])
            Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

        submitted = st.form_submit_button("🔍 Predict Loan Status", use_container_width=True)

# =================== VALIDATION ===================
if submitted:
    total_income = ApplicantIncome + SideIncome
    final_age = Age + (Loan_Amount_Term / 12)

    if Loan_Amount_Term < 12:
        st.warning("⚠️ Loan term must be at least 12 months (1 year).")
    elif Loan_Amount_Term > 480:
        st.warning("⚠️ Loan term cannot exceed 480 months (40 years).")
    elif final_age > 80:
        st.warning(f"⚠️ Age + loan duration exceeds safe limit ({final_age:.1f} years). Try reducing term.")
    else:
        payload = {
            "Gender": Gender,
            "Married": Married,
            "Dependents": Dependents,
            "Education": Education,
            "Self_Employed": Self_Employed,
            "ApplicantIncome": total_income,
            "CoapplicantIncome": CoapplicantIncome,
            "LoanAmount": LoanAmount,
            "Loan_Amount_Term": Loan_Amount_Term,
            "Credit_History": Credit_History,
            "Property_Area": Property_Area,
        }

        with st.spinner("Running prediction..."):
            try:
                response = requests.post(API_URL, json=payload)

                if response.status_code == 200:
                    result = str(response.json().get("loan_status", "N/A")).upper()

                    if result in ["Y", "APPROVED", "1"]:
                        st.markdown(
                            "<div class='result-card' style='background: linear-gradient(90deg, #00b09b, #96c93d);'>✅ Loan Approved</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        with st.spinner("💡 Checking optimal loan amount..."):
                            suggested_amount = None
                            current_amount = int(LoanAmount)
                            step = 100
                            max_attempts = 1000
                            attempt = 0

                            while current_amount > 0 and attempt < max_attempts:
                                time.sleep(0.05)
                                payload["LoanAmount"] = current_amount
                                test_response = requests.post(API_URL, json=payload)
                                if test_response.status_code == 200:
                                    test_result = str(test_response.json().get("loan_status", "N/A")).upper()
                                    if test_result in ["Y", "APPROVED", "1"]:
                                        suggested_amount = current_amount
                                        break
                                current_amount -= step
                                attempt += 1

                            if suggested_amount:
                                st.markdown(
                                    f"""
                                    <div class='result-card' style='background: linear-gradient(90deg, #cb2d3e, #ef473a);'>
                                        ❌ Loan Rejected<br><br>
                                        💡 Suggested Eligible Loan Amount: ₹{suggested_amount * 1000:,}<br>
                                        (Try applying for this or a lower amount.)
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    """
                                    <div class='result-card' style='background: linear-gradient(90deg, #cb2d3e, #ef473a);'>
                                        ❌ Loan Rejected<br><br>
                                        🚫 No eligible loan amount found.<br>
                                        📋 Try improving credit history or income before reapplying.
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                else:
                    st.error(f"API Error: {response.text}")

            except Exception as e:
                st.error(f"Connection Error: {e}")
