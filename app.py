import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="CreditWise Loan Approval", page_icon="🏦", layout="centered")

st.title("🏦 CreditWise Loan Approval Predictor")
st.write("Enter applicant details to predict whether the loan will be approved.")

MODEL_PATH = Path("models/loan_model.pkl")

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

st.info("Tip: Train and export the best model from the notebook to `models/loan_model.pkl` to enable predictions.")

# --- Basic input form (generic) ---
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        applicant_income = st.number_input("Applicant Income", min_value=0.0, value=50000.0, step=1000.0)
        coapp_income = st.number_input("Co-applicant Income", min_value=0.0, value=0.0, step=1000.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=200000.0, step=5000.0)
        loan_term = st.number_input("Loan Term (months)", min_value=1, value=360, step=12)

    with col2:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=720, step=1)
        dti = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0, step=1)
        employment_years = st.number_input("Employment Years", min_value=0, max_value=50, value=3, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    if model is None:
        st.error("Model not found. Please train and save a model as `models/loan_model.pkl` from the notebook.")
    else:
        # NOTE: Your notebook may use a different feature pipeline.
        # This simple demo expects the same order/features as training.
        X = np.array([[applicant_income, coapp_income, loan_amount, loan_term,
                       credit_score, dti, dependents, employment_years]])

        try:
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None

            if str(pred).lower() in ["yes", "1", "approved", "true"]:
                st.success("✅ Prediction: Loan Approved")
            else:
                st.warning("❌ Prediction: Loan Rejected")

            if proba is not None and len(proba) >= 2:
                st.write(f"Approval Probability: **{proba[1]*100:.2f}%**")
        except Exception as e:
            st.error("Model exists but input features don't match the training pipeline.")
            st.code(str(e))
