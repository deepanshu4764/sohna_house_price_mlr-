import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("sohna_price_model.pkl")

# Page Config
st.set_page_config(page_title="Sohna House Price Predictor", page_icon="🏡")

# Custom Styling
st.markdown("""
    <style>
        .social-icons {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .social-icons a {
            margin: 0 10px;
            text-decoration: none;
            font-size: 20px;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

# Main Title
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

# Sidebar Branding
st.sidebar.title("🔗 Connect with Me")
st.sidebar.markdown("""
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Deepanshu-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/YOUR_LINK)
[![GitHub](https://img.shields.io/badge/GitHub-Deepanshu-black?style=flat&logo=github)](https://github.com/YOUR_GITHUB)
[![Twitter](https://img.shields.io/badge/Twitter-Deepanshu-blue?style=flat&logo=twitter)](https://twitter.com/YOUR_TWITTER)
[![Instagram](https://img.shields.io/badge/Instagram-Deepanshu-purple?style=flat&logo=instagram)](https://instagram.com/YOUR_INSTAGRAM)
""", unsafe_allow_html=True)

# Social Media Links (Centered)
st.markdown("""
<div class="social-icons">
    <a href="https://www.linkedin.com/in/YOUR_LINK" target="_blank">🔗 LinkedIn</a>
    <a href="https://github.com/YOUR_GITHUB" target="_blank">💻 GitHub</a>
    <a href="https://twitter.com/YOUR_TWITTER" target="_blank">🐦 Twitter</a>
    <a href="https://instagram.com/YOUR_INSTAGRAM" target="_blank">📸 Instagram</a>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    🚀 Made with ❤️ by <b>Deepanshu</b> | Follow me for more projects!
</div>
""", unsafe_allow_html=True)
