# Code Architecture and Technical Reference

## 1. System Architecture

The project is structured with a decoupled architecture separating data transformations, model training, and runtime serving:

```
├── housing_features.py          # Shared transformation pipeline and hardware detection
├── Trainer_helper_model.ipynb   # Helper model training (spatial k-NN and Random Forest)
├── DMBI_Project_v2.ipynb        # Main model experimentation, tuning, and report generation
├── app.py                       # Interactive Streamlit application
├── housing.csv                  # Source census dataset
├── requirements.txt             # Pinned Python dependencies
├── models/                      # Serialized Joblib pipelines and JSON metadata
├── reports/                     # Numerical validation outputs (CSV and JSON)
├── figures/                     # High-resolution analytical visualizations
└── docs/                        # Complete technical and domain documentation
```

---

## 2. Shared Module: `housing_features.py`

Located at [housing_features.py](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/housing_features.py), this module serves as the single source of truth for all feature engineering and column transformations.

### 2.1 Feature Definitions and Constants

* `TARGET`: `"median_house_value"`
* `CATEGORICAL`: `["ocean_proximity"]`
* `RAW_NUMERIC`:
  `["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]`
* `ENGINEERED`:
  `["rooms_per_household", "bedrooms_per_room", "population_per_household"]`
* `NUMERIC`: `RAW_NUMERIC + ENGINEERED`
* `SKEWED`: Features transformed via $\log(1 + x)$ for linear models:
  `["total_rooms", "total_bedrooms", "population", "households", "median_income", "rooms_per_household", "bedrooms_per_room", "population_per_household"]`

### 2.2 Function API Reference

#### `add_engineered_features(df: pd.DataFrame) -> pd.DataFrame`
Computes per-household ratio features directly from raw census totals.
```python
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]
```
*Note:* Missing values in `total_bedrooms` produce `NaN` in `bedrooms_per_room`. Imputation is deliberately deferred to the subsequent `Pipeline` step to prevent data leakage across validation folds.

#### `build_preprocessor(kind: str = "tree") -> ColumnTransformer`
Constructs a scikit-learn `ColumnTransformer` suited for the model family:
* `kind="tree"`: Applies `SimpleImputer(strategy="median")` across all numeric variables and `OneHotEncoder(handle_unknown="ignore")` on `ocean_proximity`. Scaling is omitted.
* `kind="linear"`:
  - Applies `SimpleImputer(strategy="median")` $\rightarrow$ `np.log1p` $\rightarrow$ `StandardScaler()` on `SKEWED` features.
  - Applies `SimpleImputer(strategy="median")` $\rightarrow$ `StandardScaler()` on non-skewed numeric features.
  - Applies `OneHotEncoder(drop="first", handle_unknown="ignore")` to prevent multicollinearity (dummy variable trap).

#### `make_pipeline(model, kind: str = "tree") -> Pipeline`
Combines feature engineering, column preprocessing, and the estimator into an end-to-end `Pipeline`:
```python
return Pipeline([
    ("features", FunctionTransformer(add_engineered_features)),
    ("prep", build_preprocessor(kind)),
    ("model", model),
])
```

#### `split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> tuple`
Performs an 80/20 train-test split stratified on 5 discretized `median_income` bands.

#### `get_xgb_device() -> str`
Detects whether CUDA acceleration is available on the local machine:
* Verifies `xgboost.build_info().get("USE_CUDA", False)`.
* Performs a minimal test fit with `xgb.XGBRegressor(device="cuda", tree_method="hist", n_estimators=2)`.
* Returns `"cuda"` if successful, or falls back gracefully to `"cpu"`.

---

## 3. Web Application: `app.py`

Located at [app.py](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/app.py), this module provides a Streamlit user interface powered by Folium and scikit-learn.

### 3.1 Artifact Loading and Caching
```python
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(MODELS / "house_price_pipeline.joblib")
    ocean_model = joblib.load(MODELS / "ocean_model.joblib")
    income_model = joblib.load(MODELS / "income_model.joblib")
    with open(MODELS / "app_metadata.json") as f:
        meta = json.load(f)
    return pipeline, ocean_model, income_model, meta
```
`@st.cache_resource` ensures large model artifacts (such as the ~223 MB Random Forest income model) are loaded into memory once across user sessions.

### 3.2 Geospatial Coordinate Validation
To prevent extrapolation into invalid regions (e.g., clicks in the Pacific Ocean, Nevada, or Mexico), the application queries the 1-nearest neighbor distance in the k-NN ocean model:
```python
dist, _ = ocean_model.kneighbors(pd.DataFrame([[lat, lon]], columns=LOC), n_neighbors=1)
if dist[0][0] <= meta["max_click_distance_deg"]:  # 0.25 degrees (~28 km)
    location_ok = True
```
Clicks exceeding `0.25°` distance from any training census block are rejected with an informative warning.

### 3.3 Neighborhood Presets
Because users cannot know exact census block totals, three preset quartiles are provided in `meta["presets"]`:
* **Rural / small block:** 280 households, 787 population (25th percentile).
* **Typical block:** 409 households, 1,166 population (Median).
* **Dense / large block:** 605 households, 1,725 population (75th percentile).

Total rooms and bedrooms are synthesized dynamically:
$$\text{total\_rooms} = \text{rooms\_per\_house} \times \text{households}$$
$$\text{total\_bedrooms} = \text{bedrooms\_per\_house} \times \text{households}$$

---

## 4. Artifact Specifications

| File Path | Format | Size | Description |
|---|---|---|---|
| [models/house_price_pipeline.joblib](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/models/house_price_pipeline.joblib) | Joblib | ~14.9 MB | Serialized end-to-end `Pipeline` containing preprocessor and tuned XGBoost model. |
| [models/ocean_model.joblib](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/models/ocean_model.joblib) | Joblib | ~860 KB | Trained `KNeighborsClassifier` ($k=3$) mapping coordinates to ocean proximity. |
| [models/income_model.joblib](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/models/income_model.joblib) | Joblib | ~223 MB | Trained `RandomForestRegressor` estimating median income from coordinates. |
| [models/app_metadata.json](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/models/app_metadata.json) | JSON | ~818 B | Configuration schema containing preset values, feature column order, and app metrics. |
| [models/helper_metrics.json](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/models/helper_metrics.json) | JSON | ~918 B | Evaluation metrics and comparison tables for helper model selection. |

---

## 5. Developer Guide & Execution Commands

### 5.1 Environment Activation
```powershell
# In Windows PowerShell from the project root:
.venv\Scripts\activate
```

### 5.2 Training Workflow
To retrain the complete system from scratch:
1. Open and execute all cells in [Trainer_helper_model.ipynb](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/Trainer_helper_model.ipynb) to generate `ocean_model.joblib` and `income_model.joblib`.
2. Open and execute all cells in [DMBI_Project_v2.ipynb](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/DMBI_Project_v2.ipynb) to perform exploratory analysis, train the candidate models, run spatial validation, and save `house_price_pipeline.joblib`.

### 5.3 Launching the Application
```powershell
streamlit run app.py
```
To run in headless server mode:
```powershell
streamlit run app.py --server.headless true
```
The application will be served at `http://localhost:8501`.
