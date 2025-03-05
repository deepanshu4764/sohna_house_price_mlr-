import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("sohna_price_model.pkl")

st.title("🏡 Sohna House Price Predictor (MLR Model)")
st.write("📊 This app predicts house prices using a trained Multiple Linear Regression model.")

# User inputs
area = st.number_input("Enter Area (sq. ft.)", min_value=500, max_value=5000, step=50)
bedrooms = st.slider("Number of Bedrooms", 1, 6, 3)
bathrooms = st.slider("Number of Bathrooms", 1, 5, 2)
location_score = st.slider("Location Score (1-10)", 1.0, 10.0, 5.0, step=0.1)
age_of_house = st.number_input("Enter Age of House (years)", min_value=0, max_value=50, step=1)

# Predict Price
if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms, location_score, age_of_house]])
    predicted_price = model.predict(input_data)
    st.success(f"🏠 Estimated House Price: ₹{predicted_price[0]:,.2f}")

st.markdown("### 🔹 Made by **Deepanshu**")
