import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("xgb_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Loan Approval Prediction")

st.title("🏦 Loan Approval Prediction System")

st.write("Enter applicant details below:")

# Input fields

person_age = st.number_input("Age", min_value=18, max_value=100)

person_gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

person_education = st.selectbox(
    "Education",
    ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
)

person_income = st.number_input("Annual Income")

person_emp_exp = st.number_input("Employment Experience (Years)")

person_home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

loan_amnt = st.number_input("Loan Amount")

loan_intent = st.selectbox(
    "Loan Purpose",
    [
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "PERSONAL",
        "DEBTCONSOLIDATION",
        "HOMEIMPROVEMENT"
    ]
)

loan_int_rate = st.number_input("Interest Rate")

loan_percent_income = st.number_input("Loan Percent Income")

cb_person_cred_hist_length = st.number_input(
    "Credit History Length"
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850
)

previous_loan_defaults_on_file = st.selectbox(
    "Previous Loan Default",
    ["No", "Yes"]
)

# IMPORTANT:
# Replace these mappings with the actual
# label encoding values from your notebook.

gender_map = {
    "Male": 1,
    "Female": 0
}

education_map = {
    "High School": 0,
    "Associate": 1,
    "Bachelor": 2,
    "Master": 3,
    "Doctorate": 4
}

home_map = {
    "RENT": 0,
    "OWN": 1,
    "MORTGAGE": 2,
    "OTHER": 3
}

intent_map = {
    "EDUCATION": 0,
    "MEDICAL": 1,
    "VENTURE": 2,
    "PERSONAL": 3,
    "DEBTCONSOLIDATION": 4,
    "HOMEIMPROVEMENT": 5
}

default_map = {
    "No": 0,
    "Yes": 1
}

if st.button("Predict Loan Status"):

    data = np.array([[
        person_age,
        gender_map[person_gender],
        education_map[person_education],
        person_income,
        person_emp_exp,
        home_map[person_home_ownership],
        loan_amnt,
        intent_map[loan_intent],
        loan_int_rate,
        loan_percent_income,
        cb_person_cred_hist_length,
        credit_score,
        default_map[previous_loan_defaults_on_file]
    ]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Denied")