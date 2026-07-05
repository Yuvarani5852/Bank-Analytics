#!/usr/bin/env python
# coding: utf-8

# In[14]:


import streamlit as st
import pandas as pd
import joblib


# In[15]:


# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("Bank_analytics_model.pkl")

# -----------------------------
# Page Title
# -----------------------------
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Default Prediction System")

st.write("Enter customer details and click Predict.")

st.divider()


# In[16]:


# -----------------------------
# Numeric Inputs
# -----------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

avg_balance = st.number_input(
    "Average Balance",
    value=5000.0
)

n_txn = st.number_input(
    "Number of Transactions",
    value=100
)

total_credit = st.number_input(
    "Total Credit",
    value=10000.0
)

total_withdrawal = st.number_input(
    "Total Withdrawal",
    value=6000.0
)

has_card = st.selectbox(
    "Has Card",
    [0,1]
)

duration = st.number_input(
    "Loan Duration (Months)",
    value=24
)

payments = st.number_input(
    "Monthly Payment",
    value=2500.0
)


# In[17]:


# -----------------------------
# Categorical Inputs
# -----------------------------

Gender = st.selectbox(
    "Gender",
    ["Male","Female"]
)

frequency_english = st.selectbox(
    "Statement Frequency",
    [
        "Monthly Issuance",
        "Weekly Issuance",
        "After Transaction"
    ]
)

Account_type = st.selectbox(
    "Account Type",
    [
        "Standard",
        "Junior",
        "Senior"
    ]
)

loan_group = st.selectbox(
    "Loan Group",
    [
        "Small",
        "Medium",
        "Large"
    ]
)

Purpose = st.selectbox(
    "Loan Purpose",
    [
        "Housing",
        "Car",
        "Business",
        "Education",
        "Other"
    ]
)


# In[18]:


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    input_df = pd.DataFrame({

        "age":[age],
        "avg_balance":[avg_balance],
        "n_txn":[n_txn],
        "total_credit":[total_credit],
        "total_withdrawal":[total_withdrawal],
        "has_card":[has_card],
        "duration":[duration],
        "payments":[payments],
        "Gender":[Gender],
        "frequency_english":[frequency_english],
        "Account_type":[Account_type],
        "loan_group":[loan_group],
        "Purpose":[Purpose]

    })

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction")

    if prediction == 1:

        st.error("⚠ High Risk of Loan Default")

    else:

        st.success("✅ Low Risk of Loan Default")

    st.metric(
        "Probability of Default",
        f"{probability:.2%}"
    )
print("Sucessfully Completed")


# In[ ]:





# In[ ]:




