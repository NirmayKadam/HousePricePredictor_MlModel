import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_folium import st_folium
import folium
import os

# =========================
# Load Models and Scaler
# =========================
# Use a caching decorator to load models only once
@st.cache_data
def load_models_and_scalers():
    """
    Loads all the necessary model and scaler .joblib files.
    Returns the loaded objects.
    """
    try:
        main_model = joblib.load("housing_xgb_model.joblib")
        scaler = joblib.load("housing_scaler.joblib")
        ocean_model = joblib.load("ocean_model.joblib")
        ocean_encoder = joblib.load("ocean_encoder.joblib")
        income_model = joblib.load("income_model.joblib")
        return main_model, scaler, ocean_model, ocean_encoder, income_model
    except FileNotFoundError as e:
        # Return None for all if any file is missing
        return None, None, None, None, None

# Load the models
main_model, scaler, ocean_model, ocean_encoder, income_model = load_models_and_scalers()

# Check if models loaded successfully. If not, show an error and stop the app.
if not all([main_model, scaler, ocean_model, ocean_encoder, income_model]):
    st.error(
        "Error: One or more model files (.joblib) are missing. "
        "Please run the `train_helper_models_improved.py` and `model_training_improved.py` "
        "scripts to generate all necessary model files before running the app."
    )
    st.stop() # This stops the app from running further.

# Initialize session state for location validation
if 'location_valid' not in st.session_state:
    st.session_state['location_valid'] = False
if 'selected_location' not in st.session_state:
    st.session_state['selected_location'] = None

# =========================
# App Title and Description
# =========================
st.title("🏠 California Housing Price Predictor")
st.markdown("""
Welcome to the California Housing Price Predictor! This tool uses a powerful machine learning model 
to estimate the median value of a house in a given location.

**How to use:**
1.  Click on the interactive map to select a location in California.
2.  Provide some basic details about the house you're interested in.
3.  Click the 'Predict House Value' button to see the results!
""")

# =========================
# User Inputs in Sidebar
# =========================
st.sidebar.header("Step 2: Enter House Details")

# --- Inputs that a normal user would know ---
housing_age = st.sidebar.slider("Age of the House (in years)", 1, 100, 25)
num_bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
total_rooms_in_house = st.sidebar.number_input("Total Rooms in the House", min_value=1, max_value=20, value=6)
neighborhood_type = st.sidebar.selectbox(
    "Select Neighborhood Type",
    ('Suburban', 'Urban', 'Rural'),
    help="Choose the option that best describes the area."
)

# =========================
# Map for Location Input
# =========================
st.header("Step 1: Select a Location on the Map")

# Set a default location (center of California)
DEFAULT_LOCATION = [36.7783, -119.4179]
# Define the approximate bounding box for California
CALIFORNIA_BOUNDS = [(32.5, -124.5), (42, -114)]


# Create a Folium map with boundaries
m = folium.Map(
    location=DEFAULT_LOCATION, 
    zoom_start=6, 
    max_bounds=True, 
    min_zoom=5
)

# Set the visible map area to the bounds
m.fit_bounds(CALIFORNIA_BOUNDS)

# Display the map and get the last clicked point
map_data = st_folium(m, width=725, height=500)

# Get the latitude and longitude from the map click event
location_clicked = map_data and map_data.get("last_clicked")

if location_clicked:
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.session_state['selected_location'] = (lat, lon)
    st.info(f"📍 Selected Location: Latitude={lat:.4f}, Longitude={lon:.4f}")
    
    # --- Add boundary validation ---
    if (CALIFORNIA_BOUNDS[0][0] <= lat <= CALIFORNIA_BOUNDS[1][0] and
            CALIFORNIA_BOUNDS[0][1] <= lon <= CALIFORNIA_BOUNDS[1][1]):
        st.session_state['location_valid'] = True
    else:
        st.session_state['location_valid'] = False
        st.error("Selection out of bounds. Please click on a location within California.")
else:
    # This block runs on initial load and any rerun not caused by a map click.
    # We invalidate the location to ensure the user must click the map.
    st.session_state['location_valid'] = False
    st.warning("Please click on the map to select a location.")


# =========================
# Prediction Logic
# =========================
if st.button("Predict House Value", type="primary", disabled=not st.session_state.get('location_valid', False)):
    
    # On button click, retrieve the validated coordinates from session state
    lat, lon = st.session_state['selected_location']

    # --- Helper Model Predictions (Automated Inputs) ---
    # Predict ocean proximity based on coordinates
    ocean_pred_encoded = ocean_model.predict(np.array([[lat, lon]]))
    ocean_pred_label = ocean_encoder.inverse_transform(ocean_pred_encoded)[0]

    # Predict median income based on coordinates
    predicted_median_income = income_model.predict(np.array([[lat, lon]]))[0]

    # --- Feature Engineering (Translating user input for the model) ---
    # Estimate block-level stats based on neighborhood type
    if neighborhood_type == 'Urban':
        # Urban areas have higher density
        population_estimate = 2000
        households_estimate = 800
    elif neighborhood_type == 'Suburban':
        # Suburban is average density
        population_estimate = 1200
        households_estimate = 450
    else: # Rural
        # Rural is low density
        population_estimate = 700
        households_estimate = 250
    
    # Estimate total rooms on the block based on rooms in a single house
    # Assume the user's house is typical for the area
    total_rooms_block = total_rooms_in_house * households_estimate
    total_bedrooms_block = num_bedrooms * households_estimate

    # Create the DataFrame with all features in the correct order for the model
    try:
        # Calculate the ratio features that the model expects
        bedroom_ratio = total_bedrooms_block / total_rooms_block
        household_rooms = total_rooms_block / households_estimate

        input_data = pd.DataFrame({
            'longitude': [lon],
            'latitude': [lat],
            'housing_median_age': [housing_age],
            'total_rooms': [total_rooms_block],
            'total_bedrooms': [total_bedrooms_block],
            'population': [population_estimate],
            'households': [households_estimate],
            'median_income': [predicted_median_income],
            # Add the two missing ratio features
            'bedroom_ratio': [bedroom_ratio],
            'household_rooms': [household_rooms],
        })

        # --- Scale the features ---
        input_scaled = scaler.transform(input_data)

        # --- Make the final prediction ---
        prediction = main_model.predict(input_scaled)[0]

        # =========================
        # Display Results
        # =========================
        st.header("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Predicted House Value", value=f"${prediction:,.2f}")
        
        with col2:
            st.metric(label="Predicted Ocean Proximity", value=ocean_pred_label)
        
        st.success("Prediction successful! The values above are estimates based on the provided data.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

