import streamlit as st
import pickle
import numpy as np

# Load trained model
with open('SalaryPrediction.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Salary Prediction App", page_icon="💼", layout="centered")
st.title("💼 Salary Prediction App")
st.write("Enter your years of experience to estimate your predicted salary (in ₹).")

# Input field for experience
experience = st.number_input(
    label="Experience (in years):",
    min_value=0.0,
    max_value=50.0,
    value=1.0,
    step=0.5,
    help="Enter your total years of professional experience."
)

# Predict button
if st.button("🔮 Predict Salary"):
    features = np.array([[experience]])
    prediction = model.predict(features)
    salary = round(prediction[0], 2)
    st.success(f"💰 Estimated Salary: ₹{salary}")

st.markdown("---")
st.caption("Developed by Selva | Streamlit App for Salary Prediction")