# 🏦 CreditWise: Intelligent Loan Approval System

An end-to-end Machine Learning production pipeline and interactive dashboard designed to automate credit risk assessment for **SecureTrust Bank**. 

Manual verification processes are time-consuming and prone to human bias, leading to financial or business losses. This system serves as a fast, data-driven approval assistant to predict loan viability before final human review.

---

## 🚀 Problem Statement
SecureTrust Bank receives hundreds of applications daily. Manual processing creates structural bottlenecks:
* **Good customers rejected** $\rightarrow$ Immediate business loss.
* **High-risk customers approved** $\rightarrow$ Financial defaults and capital loss.

**Our Solution:** An automated ML-powered system predicting loan approval status instantly based on financial, credit, and demographic markers.

---

## 📊 Performance Summary & Visuals
* **Overall Accuracy:** 95.3%
* **Model Engine:** Random Forest Classifier (100 estimators)
* **Data Cleansing:** Fully automated pipeline handling missing entry imputation (median/mode) and standard features scaling.

*Graphs and evaluation curves (Confusion Matrix, ROC-AUC) are securely cataloged in the `assets/` directory.*

---

## 📌 Dataset & Key Features
The system processes applicant data using the following metrics to predict the target variable (`Loan_Approved`):
* **Financial Details:** Applicant Income, Co-applicant Income, DTI Ratio, Savings, Collateral Value, Loan Amount, Loan Term.
* **Credit History:** Credit Score, Existing Loans.
* **Demographics:** Employment Status, Age, Marital Status, Dependents, Property Area, Education Level, Gender, Employer Category.

---

## 📁 Repository Structure
```text
├── assets/                 # Evaluation plots (Confusion Matrix, ROC Curve)
├── data/                   # Secure vault for source datasets
├── models/                 # Saved production weights (loan_model.pkl, loan_scaler.pkl)
├── notebooks/              # Playground for exploratory data analysis (EDA)
├── src/
│   ├── preprocess.py       # Modular data engineering and encoding script
│   └── train.py           # Automated model training and evaluation script
└── app.py                  # Live Streamlit application file
⚙️ How to Run Locally
1. Initialize and Activate Virtual Environment
Bash
python -m venv venv
.\venv\Scripts\activate

2. Install Project Dependencies
Bash
pip install streamlit pandas scikit-learn joblib

3. Execute the Automated ML Training Pipeline
Bash
python src/train.py

4. Boot Up the Interactive Dashboard Demo
Bash
python -m streamlit run app.py
✅ Future Enhancements
Incorporate advanced hyperparameter tuning (GridSearchCV).

Integrate model explainability frameworks (SHAP) to debug feature weights.

Deploy the live dashboard to public cloud environments.