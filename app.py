import streamlit as st
import pandas as pd
import joblib
import os

# Set up page configurations
st.set_page_config(page_title="CreditWise - Loan Eligibility Panel", layout="centered")

# Load model and scaler safely
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/loan_model.pkl")
    scaler = joblib.load("models/loan_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    
    st.title("🏦 CreditWise Loan Approval System")
    st.markdown("Enter the applicant's financial profiles below to instantly check loan eligibility eligibility indicators.")
    st.write("---")

    # Creating user inputs matching the dataset columns
    col1, col2 = st.columns(2)
    
    with col1:
        applicant_income = st.number_input("Applicant Monthly Income ($)", min_value=0, value=5000)
        coapplicant_income = st.number_input("Co-applicant Monthly Income ($)", min_value=0, value=0)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=150000)
        
    with col2:
        age = st.number_input("Applicant Age", min_value=18, max_value=100, value=30)
        existing_loans = st.number_input("Number of Existing Loans", min_value=0, value=0)
        dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        loan_term = st.selectbox("Loan Term (Months)", [360, 180, 120, 60])

    # Let's group remaining default features behind the scenes for simplicity
    st.write("---")
    
    if st.button("Evaluate Loan Application", type="primary"):
        # Structure payload to match model feature columns
        input_data = pd.DataFrame([{
            'Applicant_Income': applicant_income,
            'Coapplicant_Income': coapplicant_income,
            'Age': age,
            'Credit_Score': credit_score,
            'Existing_Loans': existing_loans,
            'DTI_Ratio': dti_ratio,
            'Loan_Amount': loan_amount,
            'Loan_Term': loan_term,
            # Placeholder/Default flags for cat columns if your random forest requires them
            'Employment_Status_Unemployed': 0,
            'Marital_Status_Married': 1,
            'Property_Area_Urban': 1,
            'Education_Level_Graduate': 1
        }])
        
        # Keep features order matched to dataset requirements
        # (The preprocessing step generated the final trained feature index match)
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[model_features]
        
        # Scale and predict
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        
        if prediction == 1 or str(prediction).lower() == 'yes':
            st.success("🎉 **Loan Application Approved!** The applicant matches standard lower-risk margins.")
        else:
            st.error("❌ **Loan Application Declined.** Profile metrics indicate higher structural risk patterns.")

except FileNotFoundError:
    st.error("⚠️ **Artifact Mismatch:** Make sure 'loan_model.pkl' and 'loan_scaler.pkl' are inside your local `/models` directory before running the app.")