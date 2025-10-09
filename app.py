import streamlit as st
import pickle
import numpy as np
# Load your trained model
model = pickle.load(open('SalaryPrediction.pkl', 'rb'))

# App title
st.title("💼 Salary Prediction App")
st.write("Enter your details below to estimate the predicted salary (in ₹).")

# --- Input fields ---
experience = st.number_input("Experience (in years):", min_value=0.0, max_value=50.0, value=1.0, step=0.5)

# --- Prediction button ---
if st.button("🔮 Predict Salary"):
    # Convert inputs into numpy array
    features = np.array([[experience]])
    
    # Predict salary
    prediction = model.predict(features)
    salary = round(prediction[0], 2)
    
    st.success(f"💰 Estimated Salary: ₹{salary}")

# --- Footer ---
st.markdown("---")
st.caption("Developed by Selva | Streamlit App for Salary Prediction")
