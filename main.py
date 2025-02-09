# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved Linear Regression model.
model = joblib.load("linear_regression_model.pkl")

st.title("Solar Power Output Predictor")
st.write("Use the sliders below to set input values for each feature. If you leave a slider at its default, that value will be used.")

# Helper function to create slider inputs.
def slider_input(feature_name, min_val, max_val, default_val, step=None, is_int=False):
    if step is None:
        step = (max_val - min_val) / 100  # A reasonable step for continuous features.
    if is_int:
        return st.slider(feature_name, min_value=int(min_val), max_value=int(max_val), value=int(default_val), step=1)
    else:
        return st.slider(feature_name, min_value=float(min_val), max_value=float(max_val),
                         value=float(default_val), step=step, format="%.2f")

# Define slider parameters for each feature.
# The values below come from your dataset's min, max, and typical (default) values.
temperature_2_m = slider_input("temperature_2_m_above_gnd", -5.35, 34.9, 15.07)
relative_humidity = slider_input("relative_humidity_2_m_above_gnd", 7, 100, 51.36, is_int=True)
mean_sea_level_pressure = slider_input("mean_sea_level_pressure_MSL", 997.5, 1046.8, 1019.34)
total_precipitation = slider_input("total_precipitation_sfc", 0.0, 3.2, 0.03176)
snowfall_amount = slider_input("snowfall_amount_sfc", 0.0, 1.68, 0.00281)
total_cloud_cover = slider_input("total_cloud_cover_sfc", 0, 100, 34.06, is_int=True)
high_cloud_cover = slider_input("high_cloud_cover_high_cld_lay", 0, 100, 14.46, is_int=True)
medium_cloud_cover = slider_input("medium_cloud_cover_mid_cld_lay", 0, 100, 20.02, is_int=True)
low_cloud_cover = slider_input("low_cloud_cover_low_cld_lay", 0, 100, 21.37, is_int=True)
shortwave_radiation = slider_input("shortwave_radiation_backwards_sfc", 0.0, 952.3, 387.76)
wind_speed_10 = slider_input("wind_speed_10_m_above_gnd", 0.0, 20.0, 10.0)
wind_direction_10 = slider_input("wind_direction_10_m_above_gnd", 0.0, 360.0, 180.0)
wind_speed_80 = slider_input("wind_speed_80_m_above_gnd", 3.55, 25.50, 10.0)
wind_direction_80 = slider_input("wind_direction_80_m_above_gnd", 0.0, 360.0, 180.0)
wind_speed_900 = slider_input("wind_speed_900_mb", 0.0, 10.0, 5.0)
wind_direction_900 = slider_input("wind_direction_900_mb", 0.0, 360.0, 180.0)
wind_gust_10 = slider_input("wind_gust_10_m_above_gnd", 0.0, 100.0, 20.0)
angle_of_incidence = slider_input("angle_of_incidence", 0.0, 90.0, 45.0)
zenith = slider_input("zenith", 0.0, 90.0, 70.0)
azimuth = slider_input("azimuth", 0.0, 360.0, 180.0)
power_bin = slider_input("power_bin", 0, 4, 0, is_int=True)

# Build a dictionary of inputs.
input_data = {
    "temperature_2_m_above_gnd": temperature_2_m,
    "relative_humidity_2_m_above_gnd": relative_humidity,
    "mean_sea_level_pressure_MSL": mean_sea_level_pressure,
    "total_precipitation_sfc": total_precipitation,
    "snowfall_amount_sfc": snowfall_amount,
    "total_cloud_cover_sfc": total_cloud_cover,
    "high_cloud_cover_high_cld_lay": high_cloud_cover,
    "medium_cloud_cover_mid_cld_lay": medium_cloud_cover,
    "low_cloud_cover_low_cld_lay": low_cloud_cover,
    "shortwave_radiation_backwards_sfc": shortwave_radiation,
    "wind_speed_10_m_above_gnd": wind_speed_10,
    "wind_direction_10_m_above_gnd": wind_direction_10,
    "wind_speed_80_m_above_gnd": wind_speed_80,
    "wind_direction_80_m_above_gnd": wind_direction_80,
    "wind_speed_900_mb": wind_speed_900,
    "wind_direction_900_mb": wind_direction_900,
    "wind_gust_10_m_above_gnd": wind_gust_10,
    "angle_of_incidence": angle_of_incidence,
    "zenith": zenith,
    "azimuth": azimuth,
    "power_bin": power_bin
}

# Convert inputs to a DataFrame.
input_df = pd.DataFrame([input_data])

# If available, reorder columns to match the training order.
if hasattr(model, "feature_names_in_"):
    input_df = input_df[model.feature_names_in_]

st.write("Input Values:")
st.write(input_df)

# Make prediction when the button is pressed.
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted Generated Power (kW): {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
