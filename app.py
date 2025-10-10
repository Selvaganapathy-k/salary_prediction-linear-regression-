import streamlit as st
import pickle
import numpy as np

# --- Helper Functions for Serialization ---
# Note: For security and compatibility, we implement a check 
# before loading the pickle file. In a controlled environment, 
# loading a known model is safe.

def load_model_and_scaler(filepath):
    """
    Loads the model and scaler from the pickle file.
    Expects the file to contain a tuple: (model, scaler).
    """
    try:
        with open(filepath, 'rb') as f:
            loaded_data = pickle.load(f)
            
            # CRITICAL CHECK for the 'cannot unpack' error
            if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                model, scaler = loaded_data
                return model, scaler
            
            # Handle the case where only the model was saved (causing the error)
            elif not isinstance(loaded_data, tuple):
                st.error(
                    f"Error: The model file '{filepath}' returned a single object (likely just the model) "
                    f"instead of the required two objects (model, scaler). "
                    f"Please ensure your training script saves the model and scaler as a tuple: "
                    f"`pickle.dump((model, scaler), f)`."
                )
                return None, None
            else:
                st.error(f"Error: The model file '{filepath}' did not contain the expected two objects (model, scaler).")
                return None, None

    except FileNotFoundError:
        st.error(f"Error: The model file '{filepath}' was not found. Please ensure it is in the same directory.")
        return None, None
    except Exception as e:
        # Catch other potential loading errors
        st.error(f"An unexpected error occurred while loading the model: {e}")
        return None, None

# --- Configuration and Initialization ---
# Load model and scaler from the specified file
MODEL_FILEPATH = 'SalaryPrediction.pkl'
model, scaler = load_model_and_scaler(MODEL_FILEPATH)

# Check if loading was successful
if model is None or scaler is None:
    st.stop() # Stop the app execution if model failed to load

# --- Streamlit App Layout ---

# Set a slightly wider layout for better aesthetics
st.set_page_config(
    page_title="Salary Prediction App", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# App title and description with engaging icons
st.title("💼 AI-Powered Salary Predictor")
st.markdown("---")
st.info("Enter your years of experience below to get an estimated salary (in ₹).")

# Create a central container for the input and button
with st.container(border=True):
    # Input field for Experience
    experience = st.number_input(
        "Years of Experience:", 
        min_value=0.0, 
        max_value=30.0, 
        value=3.5, 
        step=0.5,
        help="Input the total number of years you have worked."
    )

    # --- Prediction Button ---
    # Centering the button and making it look professional
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Predict Estimated Salary", type="primary", use_container_width=True):
            
            # 1. Convert input into numpy array (needs to be 2D for scaler)
            features = np.array([[experience]])

            # 2. Scale the input using the loaded scaler
            features_scaled = scaler.transform(features)

            # 3. Predict salary using the loaded model
            try:
                prediction = model.predict(features_scaled)
                # The prediction returns an array, extract the first element
                salary = round(prediction[0], 2)

                # 4. Display the result
                st.balloons() # Visual celebration for the prediction
                st.success(f"💰 **Estimated Salary:** ₹{salary:,.2f}")
                st.caption(f"Based on a Linear Regression model trained on historical data.")
                
            except Exception as e:
                st.error(f"Prediction failed due to an error: {e}")


# --- Footer ---
st.markdown("---")
st.caption("Developed by Selva | Streamlit App for Salary Prediction | Model: Linear Regression")
