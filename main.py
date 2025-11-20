import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load('model.pkl')
model_features = joblib.load('model_features.pkl')

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🏠 House Price Prediction App")
st.write("Enter the house details below to predict its price.")

GrLivArea = st.number_input("Above ground living area (sq ft)", min_value=200, max_value=6000, step=50)
OverallQual = st.slider("Overall material and finish quality (1–10)", 1, 10, 5)
GarageCars = st.number_input("Number of cars that fit in garage", 0, 5, 1)
TotalBsmtSF = st.number_input("Total basement area (sq ft)", 0, 3000, 500)
YearBuilt = st.number_input("Year built", 1800, 2024, 2000)

input_data = pd.DataFrame(columns=model_features)
input_data.loc[0, ['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']] = [
    GrLivArea, OverallQual, GarageCars, TotalBsmtSF, YearBuilt
]

input_data = input_data.fillna(0)

if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")
