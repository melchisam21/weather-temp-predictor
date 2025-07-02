
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# Load model
model = xgb.XGBRegressor()
# It's better to load the model using the file name directly if it was saved using save_model
model.load_model("xgb_model.json")

# App title
st.title("🌤️ Weather Max Temperature Predictor")

st.write("Enter the current weather conditions to predict today's max temperature (°C).")

# Sidebar inputs
# Based on the feature importance and the data used for training (precipitation, temp_min, wind)
rainfall = st.number_input("Precipitation (mm)", 0.0, 100.0, float(X_train['precipitation'].mean()))
temp_min = st.number_input("Minimum Temperature (°C)", float(X_train['temp_min'].min()), float(X_train['temp_min'].max()), float(X_train['temp_min'].mean()))
wind = st.number_input("Wind (km/h)", float(X_train['wind'].min()), float(X_train['wind'].max()), float(X_train['wind'].mean()))

# Note: Removed inputs not used in the trained model (Humidity, Pressure, Temp at 9am/3pm, RainToday, Wind Direction)
# These features were not included in the X dataframe used for training the XGBoost model.

# Encode inputs
input_data = {
    "precipitation": rainfall,
    "temp_min": temp_min,
    "wind": wind,
}

input_df = pd.DataFrame([input_data])

# Ensure column order matches training data if necessary (though XGBoost handles this well)
# input_df = input_df[['precipitation', 'temp_min', 'wind']] # Example if strict order needed

# Predict
if st.button("Predict Max Temperature"):
    prediction = model.predict(input_df)[0]
    st.success(f"🌡️ Predicted Max Temperature: **{prediction:.2f}°C**")
