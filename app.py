import streamlit as st
import pickle
import numpy as np

# Load both model and scaler
with open('SalaryPrediction.pkl', 'rb') as f:
    model, scaler = pickle.load(f)

# App title
st.title("💼 Salary Prediction App")
st.write("Enter your details below to estimate the predicted salary (in ₹).")

# --- Input fields ---
experience = st.number_input("Experience (in years):", min_value=0.0, max_value=50.0, value=1.0, step=0.5)

# --- Prediction button ---
if st.button("🔮 Predict Salary"):
    # Convert input into numpy array
    features = np.array([[experience]])

    # Scale the input
    features_scaled = scaler.transform(features)

    # Predict salary
    prediction = model.predict(features_scaled)
    salary = round(prediction[0], 2)

    st.success(f"💰 Estimated Salary: ₹{salary}")

# --- Footer ---
st.markdown("---")
st.caption("Developed by Selva | Streamlit App for Salary Prediction")
