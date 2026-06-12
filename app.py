import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("model/trained_model.h5")
scaler = joblib.load("model/scaler.pkl")

st.title("Bank Customer Churn Prediction")

# Inputs
customer_id = st.number_input("Customer ID", value=15634602)

credit_score = st.number_input("Credit Score", 300, 900, 600)
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure", 0, 20, 5)

balance = st.number_input("Balance", value=0.0)
products_number = st.number_input("Products Number", 1, 4, 1)

credit_card = st.selectbox("Credit Card", [0, 1])
active_member = st.selectbox("Active Member", [0, 1])

estimated_salary = st.number_input("Estimated Salary", value=50000.0)

country = st.selectbox(
    "Country",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

# One-Hot Encoding
country_Germany = 1 if country == "Germany" else 0
country_Spain = 1 if country == "Spain" else 0
gender_Male = 1 if gender == "Male" else 0

if st.button("Predict Churn"):

    data = pd.DataFrame([[
        customer_id,
        credit_score,
        age,
        tenure,
        balance,
        products_number,
        credit_card,
        active_member,
        estimated_salary,
        country_Germany,
        country_Spain,
        gender_Male
    ]], columns=[
        'customer_id',
        'credit_score',
        'age',
        'tenure',
        'balance',
        'products_number',
        'credit_card',
        'active_member',
        'estimated_salary',
        'country_Germany',
        'country_Spain',
        'gender_Male'
    ])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0][0]

    st.write("Churn Probability:", round(float(prediction), 4))

    if prediction > 0.5:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer is likely to Stay")