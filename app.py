import json
from pathlib import Path

import folium
import joblib
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import housing_features  # noqa: F401  (needed so joblib can unpickle the saved pipeline)

MODELS, REPORTS, FIG = Path("models"), Path("reports"), Path("figures")
LOC = ["latitude", "longitude"]

st.set_page_config(page_title="California House Price Predictor", page_icon="🏠", layout="wide")


# =========================
# Load models and metadata
# =========================
@st.cache_resource
def load_artifacts():
    """Load the full price pipeline, the two helper models and the app metadata once."""
    pipeline = joblib.load(MODELS / "house_price_pipeline.joblib")
    ocean_model = joblib.load(MODELS / "ocean_model.joblib")
    income_model = joblib.load(MODELS / "income_model.joblib")
    with open(MODELS / "app_metadata.json") as f:
        meta = json.load(f)
    return pipeline, ocean_model, income_model, meta


try:
    pipeline, ocean_model, income_model, meta = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files are missing. Run all cells in `Trainer_helper_model.ipynb` first, "
        "then in `DMBI_Project_v2.ipynb`; together they create the files in `models/`."
    )
    st.stop()

# =========================
# Title and description
# =========================
st.title("🏠 California House Price Predictor")
st.markdown(
    "Estimate the **median house value of a California neighbourhood** (census block group) "
    f"with a {meta['best_model']} model trained on the 1990 census.\n\n"
    "1. Enter basic details in the sidebar.\n"
    "2. Click a location on the map.\n"
    "3. Press **Predict house value**."
)

# =========================
# User inputs (sidebar)
# =========================
st.sidebar.header("Step 1: House and area details")
housing_age = st.sidebar.slider("Typical age of houses in the area (years)", 1, 52, 25,
                                help="The training data caps age at 52 years.")
rooms = st.sidebar.number_input("Rooms per house", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
bedrooms = st.sidebar.number_input("Bedrooms per house", min_value=0.5, max_value=8.0, value=2.0, step=0.5)
block_type = st.sidebar.selectbox(
    "Neighbourhood size", list(meta["presets"].keys()), index=1,
    help="Sets households and population of the block group from the dataset's quartiles.",
)
if bedrooms > rooms:
    st.sidebar.error("Bedrooms cannot exceed total rooms.")

# =========================
# Map input
# =========================
st.header("Step 2: Select a location")
m = folium.Map(location=[37.2, -119.5], zoom_start=6, min_zoom=5, max_bounds=True,
               min_lat=30, max_lat=44, min_lon=-127, max_lon=-111)
map_data = st_folium(m, width=900, height=500, returned_objects=["last_clicked"])

clicked = map_data.get("last_clicked") if map_data else None
location_ok = False
if clicked:
    lat, lon = clicked["lat"], clicked["lng"]
    # Accept only clicks close to real training data: this rejects the ocean, Nevada and Mexico,
    # which a simple bounding box would let through.
    dist, _ = ocean_model.kneighbors(pd.DataFrame([[lat, lon]], columns=LOC), n_neighbors=1)
    if dist[0][0] <= meta["max_click_distance_deg"]:
        location_ok = True
        st.info(f"📍 Selected: latitude {lat:.4f}, longitude {lon:.4f}")
    else:
        st.error(f"({lat:.3f}, {lon:.3f}) is too far from any California neighbourhood in the data. "
                 "Please click on land in California.")
else:
    st.warning("Click on the map to select a location.")

# =========================
# Prediction
# =========================
if st.button("Predict house value", type="primary", disabled=not location_ok or bedrooms > rooms):
    loc_df = pd.DataFrame([[lat, lon]], columns=LOC)
    ocean_label = ocean_model.predict(loc_df)[0]
    income = float(income_model.predict(loc_df)[0])

    preset = meta["presets"][block_type]
    households = preset["households"]
    # Raw block-group totals, exactly as in the training data; the saved pipeline does all
    # feature engineering, imputation and encoding itself.
    row = pd.DataFrame([{
        "longitude": lon,
        "latitude": lat,
        "housing_median_age": housing_age,
        "total_rooms": rooms * households,
        "total_bedrooms": bedrooms * households,
        "population": preset["population"],
        "households": households,
        "median_income": income,
        "ocean_proximity": ocean_label,
    }])[meta["input_columns"]]
    prediction = float(pipeline.predict(row)[0])

    st.header("Prediction")
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated median house value", f"${prediction:,.0f}")
    c2.metric("Estimated area median income", f"${income * 10000:,.0f}")
    c3.metric("Estimated ocean proximity", ocean_label)

    app = meta["app_metrics"]
    # "\$" stops Streamlit's markdown from reading dollar signs as LaTeX math
    st.caption(
        f"Typical error of this app: ±\\${app['MAE']:,.0f} (MAE), R² {app['R2']:.2f}. Income and ocean proximity "
        "are estimated from the location, which adds error compared with the notebook model "
        f"(R² {meta['test_metrics']['R2']:.2f}). Values are in 1990 dollars, and the training data caps "
        "house values at \\$500,001, so estimates near that level are understated."
    )
    if prediction >= 480000:
        st.warning("This estimate is near the \\$500k cap in the training data; the true value is likely higher.")

# =========================
# Model insights
# =========================
with st.expander("Model insights and limitations"):
    comp = REPORTS / "model_comparison.csv"
    if comp.exists():
        df = pd.read_csv(comp, index_col=0)[["RMSE mean", "R2 mean", "test MAE", "test RMSE", "test R2"]]
        st.dataframe(df.style.format("{:,.3f}", subset=["R2 mean", "test R2"])
                     .format("{:,.0f}", subset=["RMSE mean", "test MAE", "test RMSE"]))
    if (FIG / "dashboard.png").exists():
        st.image(str(FIG / "dashboard.png"))
    st.markdown(
        "- Trained on **1990** census block groups: this is a teaching tool, not a valuation.\n"
        "- Location-based pricing can reproduce historical discrimination (redlining). "
        "Do not use it for lending, insurance or rental decisions about individuals."
    )
