import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(page_title="Loan Default Prediction", layout="centered")

# Load the trained pipeline
@st.cache_resource
def load_model():
    model = joblib.load("loan_default_pipeline.pkl")
    return model

clf = load_model()

st.title("🏦 Loan Default Risk Predictor")
st.write(
    "This app uses a machine learning model trained on the Home Credit dataset "
    "to estimate the probability that a loan applicant will **default**."
)

st.markdown("---")

# --- User Inputs (these must match the features we trained on) ---

st.subheader("👤 Applicant Information")

# Numeric inputs
income = st.number_input(
    "Annual Income (AMT_INCOME_TOTAL) in local currency",
    min_value=0.0,
    value=150000.0,
    step=1000.0,
)

credit_amount = st.number_input(
    "Total Credit Amount (AMT_CREDIT)",
    min_value=0.0,
    value=500000.0,
    step=1000.0,
)

annuity = st.number_input(
    "Annuity (Monthly Loan Payment) – AMT_ANNUITY",
    min_value=0.0,
    value=25000.0,
    step=500.0,
)

goods_price = st.number_input(
    "Goods Price (AMT_GOODS_PRICE)",
    min_value=0.0,
    value=500000.0,
    step=1000.0,
)

age_years = st.slider("Age (years)", min_value=18, max_value=80, value=35)
years_employed = st.slider(
    "Years Employed (can be 0 if unemployed)", min_value=0, max_value=50, value=5
)

children = st.number_input(
    "Number of Children (CNT_CHILDREN)", min_value=0, max_value=10, value=0, step=1
)

family_members = st.number_input(
    "Total Family Members (CNT_FAM_MEMBERS)", min_value=1.0, max_value=20.0, value=2.0
)

st.markdown("---")

st.subheader("🏠 Contract & Personal Profile")

contract_type = st.selectbox(
    "Contract Type (NAME_CONTRACT_TYPE)",
    options=["Cash loans", "Revolving loans"],
)

gender = st.selectbox(
    "Gender (CODE_GENDER)",
    options=["M", "F"],
)

family_status = st.selectbox(
    "Family Status (NAME_FAMILY_STATUS)",
    options=[
        "Single / not married",
        "Married",
        "Civil marriage",
        "Separated",
        "Widow",
    ],
)

income_type = st.selectbox(
    "Income Type (NAME_INCOME_TYPE)",
    options=[
        "Working",
        "Commercial associate",
        "Pensioner",
        "State servant",
        "Unemployed",
        "Student",
        "Businessman",
        "Maternity leave",
    ],
)

education_type = st.selectbox(
    "Education Level (NAME_EDUCATION_TYPE)",
    options=[
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ],
)

st.markdown("---")

# Convert age and employment years to DAYS_BIRTH and DAYS_EMPLOYED like in the original dataset
days_birth = -age_years * 365  # negative number in the dataset
days_employed = -years_employed * 365  # negative; if 0 years, it's close to 0

# Create a single-row DataFrame with the exact columns the model expects
input_data = pd.DataFrame(
    {
        "AMT_INCOME_TOTAL": [income],
        "AMT_CREDIT": [credit_amount],
        "AMT_ANNUITY": [annuity],
        "AMT_GOODS_PRICE": [goods_price],
        "DAYS_BIRTH": [days_birth],
        "DAYS_EMPLOYED": [days_employed],
        "CNT_CHILDREN": [children],
        "CNT_FAM_MEMBERS": [family_members],
        "NAME_CONTRACT_TYPE": [contract_type],
        "CODE_GENDER": [gender],
        "NAME_FAMILY_STATUS": [family_status],
        "NAME_INCOME_TYPE": [income_type],
        "NAME_EDUCATION_TYPE": [education_type],
    }
)

st.subheader("📊 Model Input Preview")
st.dataframe(input_data)

if st.button("Predict Default Risk"):
    # Get prediction
    proba_default = clf.predict_proba(input_data)[:, 1][0]
    proba_no_default = 1 - proba_default

    st.markdown("## 🔮 Prediction Result")

    st.write(f"**Probability of Default:** `{proba_default:.2%}`")
    st.write(f"**Probability of Paying Back:** `{proba_no_default:.2%}`")

    # Simple decision threshold
    if proba_default >= 0.5:
        st.error(
            "⚠️ The model considers this applicant **HIGH RISK** of default. "
            "A manual review or stricter conditions may be required."
        )
    else:
        st.success(
            "✅ The model considers this applicant **LOW RISK** of default. "
            "They are more likely to repay the loan."
        )

    st.caption(
        "Note: This is a demonstration model trained on historical data. "
        "It should not be used as a sole decision-making tool in production."
    )
