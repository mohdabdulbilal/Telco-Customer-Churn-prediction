import streamlit as st
import pickle
import numpy as np

# ================= LOAD MODEL AND SCALER =================

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ================= PAGE CONFIG =================

# ===== CUSTOM BACKGROUND & MESSAGES =====
st.markdown(
    """
    <style>
    /* Gradient background for the whole app */
    .stApp {
        background: linear-gradient(to right, #74ebd5, #ACB6E5);
        background-attachment: fixed;
    }

    /* Card-like sections with semi-transparent white */
    .css-18e3th9 {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 15px;
        border-radius: 10px;
    }

    /* Success message styling */
    .stAlert-success {
        background-color: #28a745 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
    }

    /* Error message styling */
    .stAlert-error {
        background-color: #dc3545 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
    }

    /* Optional: Animated gradient for fun */
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .stApp {
        background: linear-gradient(-45deg, #74ebd5, #ACB6E5, #FFDEE9, #B5FFFC);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Telco Customer Churn Prediction")
st.markdown(
    "Predict whether a customer is likely to **churn or stay** based on service usage and billing details."
)

st.markdown("---")

# ================= CUSTOMER INFO =================
st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])

with col2:
    Partner = st.selectbox("Partner", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])

with col3:
    tenure = st.number_input("Tenure (months)", min_value=0)

# ================= SERVICES =================
st.markdown("---")
st.subheader("📞 Services & Internet")

col4, col5, col6 = st.columns(3)

with col4:
    MultipleLines = st.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )
    InternetService = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )

with col5:
    OnlineSecurity = st.selectbox(
        "Online Security", ["No", "Yes", "No internet service"]
    )
    OnlineBackup = st.selectbox(
        "Online Backup", ["No", "Yes", "No internet service"]
    )

with col6:
    DeviceProtection = st.selectbox(
        "Device Protection", ["No", "Yes", "No internet service"]
    )
    TechSupport = st.selectbox(
        "Tech Support", ["No", "Yes", "No internet service"]
    )

col7, col8 = st.columns(2)

with col7:
    StreamingTV = st.selectbox(
        "Streaming TV", ["No", "Yes", "No internet service"]
    )

with col8:
    StreamingMovies = st.selectbox(
        "Streaming Movies", ["No", "Yes", "No internet service"]
    )

# ================= BILLING =================
st.markdown("---")
st.subheader("💳 Billing Information")

col9, col10, col11 = st.columns(3)

with col9:
    Contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )

with col10:
    PaperlessBilling = st.selectbox(
        "Paperless Billing", ["No", "Yes"]
    )

with col11:
    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

col12, col13 = st.columns(2)

with col12:
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)

with col13:
    TotalCharges = st.number_input("Total Charges", min_value=0.0)

# ================= PREPROCESSING =================

# Gender
gender = 1 if gender == "Male" else 0

# Binary mapping
binary_map = {"Yes": 1, "No": 0}
Partner = binary_map[Partner]
Dependents = binary_map[Dependents]
PaperlessBilling = binary_map[PaperlessBilling]

# Internet-related fix
def internet_fix(val):
    return 0 if val in ["No", "No internet service"] else 1

OnlineSecurity = internet_fix(OnlineSecurity)
OnlineBackup = internet_fix(OnlineBackup)
DeviceProtection = internet_fix(DeviceProtection)
TechSupport = internet_fix(TechSupport)
StreamingTV = internet_fix(StreamingTV)
StreamingMovies = internet_fix(StreamingMovies)

# Derived columns
NoInternetService = 1 if InternetService == "No" else 0
NoPhoneService = 1 if MultipleLines == "No phone service" else 0

# MultipleLines
MultipleLines = 0 if MultipleLines in ["No", "No phone service"] else 1

# Manual label encoding
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}

InternetService = internet_map[InternetService]
Contract = contract_map[Contract]
PaymentMethod = payment_map[PaymentMethod]

# ================= MODEL INPUT =================
input_data = np.array([[  
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    MonthlyCharges,
    TotalCharges,
    NoInternetService,
    NoPhoneService
]])

input_data_scaled = scaler.transform(input_data)

# ================= PREDICTION =================
st.markdown("---")
st.subheader("🔍 Prediction Result")

if st.button("🚀 Predict Churn", use_container_width=True):
    prediction = model.predict(input_data_scaled)[0]

    if prediction == 1:
        st.error("❌ **Customer is likely to churn**")
    else:
        st.success("✅ **Customer is likely to stay**")

st.caption("📌 Telco Customer Churn Prediction | ML Demo App")
