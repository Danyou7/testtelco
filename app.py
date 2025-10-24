import streamlit as st
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# ==============================
# Load model
# ==============================
model = CatBoostClassifier()
model.load_model("classification_catboost_tuned.cbm")

st.set_page_config(page_title="Prediksi Customer Churn", page_icon="📊", layout="centered")

st.title("📊 Prediksi Customer Churn Menggunakan CatBoost")
st.write("Masukkan data pelanggan untuk memprediksi apakah pelanggan berpotensi **Churn (berhenti)** atau **Tidak Churn**.")

# ==============================
# Input Kategorikal
# ==============================
st.header("🧩 Data Kategorikal")

gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
online_backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])
device_protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
tech_support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# ==============================
# Input Numerik
# ==============================
st.header("🔢 Data Numerik")

tenure_months = st.number_input("Tenure Months", min_value=0, step=1)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=0.1)
total_charges = st.number_input("Total Charges", min_value=0.0, step=0.1)
cltv = st.number_input("Customer Lifetime Value (CLTV)", min_value=0.0, step=1.0)

# ==============================
# Mapping nilai kategorikal
# ==============================
mapping = {
    "Gender": {"Female": 0, "Male": 1},
    "Senior Citizen": {"No": 0, "Yes": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "Phone Service": {"No": 0, "Yes": 1},
    "Multiple Lines": {"No": 0, "No phone service": 1, "Yes": 2},
    "Internet Service": {"DSL": 0, "Fiber optic": 1, "No": 2},
    "Online Security": {"No": 0, "No internet service": 1, "Yes": 2},
    "Online Backup": {"No": 0, "No internet service": 1, "Yes": 2},
    "Device Protection": {"No": 0, "No internet service": 1, "Yes": 2},
    "Tech Support": {"No": 0, "No internet service": 1, "Yes": 2},
    "Streaming TV": {"No": 0, "No internet service": 1, "Yes": 2},
    "Streaming Movies": {"No": 0, "No internet service": 1, "Yes": 2},
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "Paperless Billing": {"No": 0, "Yes": 1},
    "Payment Method": {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }
}

# ==============================
# Buat DataFrame sesuai urutan model
# ==============================
input_data = pd.DataFrame([[
    mapping["Gender"][gender],
    mapping["Senior Citizen"][senior_citizen],
    mapping["Partner"][partner],
    mapping["Dependents"][dependents],
    tenure_months,
    mapping["Phone Service"][phone_service],
    mapping["Multiple Lines"][multiple_lines],
    mapping["Internet Service"][internet_service],
    mapping["Online Security"][online_security],
    mapping["Online Backup"][online_backup],
    mapping["Device Protection"][device_protection],
    mapping["Tech Support"][tech_support],
    mapping["Streaming TV"][streaming_tv],
    mapping["Streaming Movies"][streaming_movies],
    mapping["Contract"][contract],
    mapping["Paperless Billing"][paperless_billing],
    mapping["Payment Method"][payment_method],
    monthly_charges,
    total_charges,
    cltv
]], columns=[
    'Gender', 'Senior Citizen', 'Partner', 'Dependents',
    'Tenure Months', 'Phone Service', 'Multiple Lines', 'Internet Service',
    'Online Security', 'Online Backup', 'Device Protection', 'Tech Support',
    'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing',
    'Payment Method', 'Monthly Charges', 'Total Charges', 'CLTV'
])

# ==============================
# Prediksi
# ==============================
if st.button("🔍 Prediksi Churn"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("📢 Hasil Prediksi:")
    if pred == 1:
        st.error(f"Pelanggan **BERPOTENSI CHURN** dengan probabilitas {prob:.2%}")
    else:
        st.success(f"Pelanggan **TIDAK CHURN** dengan probabilitas {prob:.2%}")

    st.caption("Model: CatBoost Tuned | Dibuat oleh Danu Tirta")
