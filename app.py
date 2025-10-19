import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Streamlit App Layout and Logic ---
st.set_page_config(
    page_title="Sales Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("💰 Advertising Sales Predictor")
st.markdown("Use the sliders below to estimate sales based on advertising spend across three major platforms.")

# Load the model from "model-reg-67130701716.pkl"
try:
    loaded_model = pickle.load(open('model-reg-67130701716.pkl', 'rb'))
    st.success("Regression model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# --- User Input Widgets (Features) ---
st.sidebar.header("Input Advertising Spend ($)")

# Define input parameters using Streamlit sliders
youtube_spend = st.sidebar.slider("YouTube Budget ($)", 0, 500, 100, step=1)
tiktok_spend = st.sidebar.slider("TikTok Budget ($)", 0, 200, 50, step=1)
instagram_spend = st.sidebar.slider("Instagram Budget ($)", 0, 200, 50, step=1)


# Create a new DataFrame with user inputs
input_data = pd.DataFrame({
    'youtube': [youtube_spend],
    'tiktok': [tiktok_spend],
    'instagram': [instagram_spend]
})

st.subheader("Current Input Budget")
st.dataframe(input_data, use_container_width=True)


# --- Prediction Button and Output ---
if st.button("Predict Estimated Sales", type="primary"):
    with st.spinner('Calculating sales prediction...'):
        try:
            # Make predictions on the new DataFrame
            predicted_sales = loaded_model.predict(input_data)
            estimated_sales = predicted_sales[0]

            st.markdown("---")

            st.metric(
                label="Estimated Sales",
                value=f"${estimated_sales:,.2f}",
                delta="Based on current budget allocation"
            )

            st.balloons()

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Display model coefficients if the model is loaded successfully
if 'loaded_model' in locals():
    st.sidebar.subheader("Model Coefficients")
    features = ['youtube', 'tiktok', 'instagram']
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': loaded_model.coef_
    })
    st.sidebar.dataframe(coef_df, hide_index=True)
    st.sidebar.markdown(f"**Intercept (Base Sales):** `{loaded_model.intercept_:.2f}`") 
