
import streamlit as st
import pandas as pd
import xgboost as xgb

# Load model
model = xgb.XGBRegressor()
model.load_model("xgb_model.json")

# App title
st.title("🌤️ Weather Max Temperature Predictor")
st.write("Enter the current weather conditions to predict today's max temperature (°C).")

# Sidebar inputs
rainfall = st.number_input("Rainfall (mm)", 0.0, 100.0, 0.0)
humidity9am = st.slider("Humidity at 9am (%)", 0, 100, 60)
humidity3pm = st.slider("Humidity at 3pm (%)", 0, 100, 50)
pressure9am = st.number_input("Pressure at 9am (hPa)", 980.0, 1040.0, 1010.0)
pressure3pm = st.number_input("Pressure at 3pm (hPa)", 980.0, 1040.0, 1012.0)
temp9am = st.slider("Temperature at 9am (°C)", 0.0, 40.0, 20.0)
temp3pm = st.slider("Temperature at 3pm (°C)", 0.0, 45.0, 25.0)

rain_today = st.selectbox("Did it rain today?", ["No", "Yes"])
wind_dir = st.selectbox("Wind Direction at 3pm", ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

# Encode inputs
input_data = {
    "Rainfall": rainfall,
    "Humidity9am": humidity9am,
    "Humidity3pm": humidity3pm,
    "Pressure9am": pressure9am,
    "Pressure3pm": pressure3pm,
    "Temp9am": temp9am,
    "Temp3pm": temp3pm,
    "RainToday": 1 if rain_today == "Yes" else 0,
}

# One-hot encoding for wind direction
wind_directions = ['NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']  # 'N' is the dropped base
for direction in wind_directions:
    input_data[f"WindDir3pm_{direction}"] = 1 if wind_dir == direction else 0

input_df = pd.DataFrame([input_data])

# Predict
if st.button("Predict Max Temperature"):
    prediction = model.predict(input_df)[0]
    st.success(f"🌡️ Predicted Max Temperature: **{prediction:.2f}°C**")
