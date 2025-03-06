import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("sohna_price_model.pkl")

# Social media icons at the top right (High-Quality Professional Icons)
st.markdown(
    """
    <style>
    .icon-container {
        position: absolute;
        top: 10px;
        right: 10px;
    }
    .icon-container a {
        margin-left: 10px;
    }
    </style>
    <div class="icon-container">
        <a href="https://github.com/deepanshu4764" target="_blank">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="35" alt="GitHub"/>
        </a>
        <a href="https://www.linkedin.com/in/deepanshu4764" target="_blank">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="35" alt="LinkedIn"/>
        </a>
        <a href="https://twitter.com/deepanshu4764" target="_blank">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/twitter/twitter-original.svg" width="35" alt="Twitter"/>
        </a>
        <a href="https://www.instagram.com/deepanshu4764" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png" width="35" alt="Instagram"/>
        </a>
        <a href="https://www.youtube.com/@deepanshu4764" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_social_white_squircle_%282017%29.svg" width="35" alt="YouTube"/>
        </a>
        <a href="https://chat.whatsapp.com/deepanshu4764" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg" width="35" alt="WhatsApp Community"/>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

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
