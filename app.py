
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Function to Create a Mock Model (Simulating the .pkl file content) ---
# This function trains a simple Linear Regression model and serializes it
# so the app can "load" it without needing an external file.
@st.cache_resource
def get_mock_model_bytes():
    # 1. Create a tiny, predictable mock dataset
    data = {
        'youtube': [100, 200, 50, 250],
        'tiktok': [10, 20, 5, 25],
        'instagram': [5, 15, 2, 20],
        # Sales are calculated as 0.05*YouTube + 0.1*TikTok + 0.3*Instagram + 5 (Intercept)
        'sales': [10.5, 17.5, 7.5, 21.0]
    }
    df = pd.DataFrame(data)
    X = df[['youtube', 'tiktok', 'instagram']]
    y = df['sales']

    # 2. Train the model
    mock_model = LinearRegression()
    mock_model.fit(X, y)

    # 3. Serialize the model using pickle (simulating the saved .pkl file)
    model_bytes = pickle.dumps(mock_model)
    return model_bytes

# --- Streamlit App Layout and Logic ---

st.set_page_config(
    page_title="Sales Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("💰 Advertising Sales Predictor")
st.markdown("Use the sliders below to estimate sales based on advertising spend across three major platforms.")

# Step 1: Simulate loading the model from "model-reg-67130701716.pkl"
# In a real app, you'd use: loaded_model = pickle.load(open('model-reg-67130701716.pkl', 'rb'))
try:
    model_bytes = get_mock_model_bytes()
    loaded_model = pickle.loads(model_bytes)
    st.success("Regression model loaded successfully (Simulated .pkl load).")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# --- User Input Widgets (Features) ---

st.sidebar.header("Input Advertising Spend ($)")

# Define input parameters using Streamlit sliders
youtube_spend = st.sidebar.slider("YouTube Budget ($)", 0, 500, 50)
tiktok_spend = st.sidebar.slider("TikTok Budget ($)", 0, 100, 50)
instagram_spend = st.sidebar.slider("Instagram Budget ($)", 0, 100, 50)


# Step 2: Create a new DataFrame with user inputs
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
            # Step 3: Make predictions on the new DataFrame
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

st.markdown("---")
st.caption("Note: This app uses a mocked Linear Regression model for demonstration.")

# Display model coefficients if the model is loaded successfully
if 'loaded_model' in locals():
    st.sidebar.subheader("Model Coefficients")
    # Assuming the mock model has these feature names
    features = ['youtube', 'tiktok', 'instagram']
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': loaded_model.coef_
    })
    st.sidebar.dataframe(coef_df, hide_index=True)
    st.sidebar.markdown(f"**Intercept (Base Sales):** `{loaded_model.intercept_:.3f}`")
