
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# Load model
model = xgb.XGBRegressor()
model.load_model("xgb_model.json")

# App title
st.title("🌤️ Weather Max Temperature Predictor")

st.write("Enter the current weather conditions to predict today's max temperature (°C).")

# Sidebar inputs (replaced dynamic values with safe defaults)
rainfall = st.number_input("Precipitation (mm)", 0.0, 100.0, 5.0)
temp_min = st.number_input("Minimum Temperature (°C)", -5.0, 35.0, 15.0)
wind = st.number_input("Wind (km/h)", 0.0, 100.0, 20.0)

# Prepare input data
input_data = {
    "precipitation": rainfall,
    "temp_min": temp_min,
    "wind": wind,
}

input_df = pd.DataFrame([input_data])

# Predict
if st.button("Predict Max Temperature"):
    prediction = model.predict(input_df)[0]
    st.success(f"🌡️ Predicted Max Temperature: **{prediction:.2f}°C**")
