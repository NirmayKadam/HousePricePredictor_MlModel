# California House Price Predictor

An end-to-end Machine Learning and Spatial Data Science project (DMBI) that predicts the **median house value of California census block groups**. The project benchmarks Ordinary Least Squares **Linear Regression** against **Random Forest** and **XGBoost** (GPU-accelerated with CUDA) and deploys the best-performing model in an interactive **Streamlit** geospatial application.

---

## Documentation Suite

Detailed technical, mathematical, and architectural documentation is available in the [`docs/`](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/docs) directory:

* **[Documentation Hub (`docs/README.md`)](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/docs/README.md):** Full navigation index and guide.
* **[Research and Dataset (`docs/research_and_dataset.md`)](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/docs/research_and_dataset.md):** Census block group granularity, Pace & Barry (1997) origin, distributions, $500,001 censoring, and ethical considerations surrounding algorithmic redlining.
* **[ML Training and Process (`docs/ml_training_and_process.md`)](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/docs/ml_training_and_process.md):** Data leakage prevention, income-stratified splitting, OLS from first principles (Normal Equations and SVD), GPU-accelerated XGBoost tuning, and coordinate helper models.
* **[Results and Insights (`docs/results_and_insights.md`)](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/docs/results_and_insights.md):** Cross-validation vs test benchmarks, paired t-test ($p < 0.001$), spatial holdout cross-validation (`Spatial GroupKFold`), residual diagnostics, and feature importance.
* **[Code Architecture (`docs/code_architecture.md`)](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/docs/code_architecture.md):** Codebase structure, `housing_features.py` API reference, `app.py` caching and geodesic validation, and serialized model schemas.

---

## Dataset Overview

| Attribute | Specification |
|---|---|
| **Data File** | [`housing.csv`](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/housing.csv): 20,640 rows × 10 columns |
| **Origin** | 1990 U.S. Census, compiled by R. K. Pace & R. Barry, *"Sparse Spatial Autoregressions"*, Statistics & Probability Letters 33 (1997) 291–297; distributed via StatLib (CMU) |
| **Variant** | From A. Géron, *Hands-On Machine Learning* (adds `ocean_proximity` and 207 missing `total_bedrooms`) |
| **Unit of Analysis** | One **census block group** (~600–3,000 residents), **not** an individual home |
| **Target Variable** | `median_house_value` (USD, right-censored at $500,001) |

---

## Benchmark Results

Evaluated on the held-out test set (4,128 block groups) after 5-fold cross-validation on the 16,512 training instances:

| Model | 5-Fold CV RMSE (mean ± std) | Test MAE | Test RMSE | Test $R^2$ | Overfit Gap |
|---|---|---|---|---|---|
| **Baseline (Mean)** | $115,699 ± 1,068 | $90,807 | $114,165 | 0.000 | 0.000 |
| **Linear Regression (OLS)** | $67,831 ± 742 | $50,542 | $67,807 | 0.647 | 0.012 |
| **Linear Regression (Log Target)** | $66,027 ± 972 | $44,974 | $79,670 | 0.513 | 0.165 |
| **Random Forest (CPU)** | $49,153 ± 877 | $31,111 | $46,826 | 0.832 | 0.144 |
| **XGBoost (GPU CUDA)** | **$45,053 ± 964** | **$28,219** | **$42,736** | **0.860** | **0.123** |

### Key Experimental Findings
1. **Statistical Superiority:** XGBoost outperforms Random Forest across every cross-validation fold (paired two-tailed t-test: $t = 23.52, p < 0.001$), reducing test RMSE by 37.0% relative to OLS.
2. **Spatial Autocorrelation:** Under 0.5° coordinate holdout (`Spatial GroupKFold`), tree models experience an $R^2$ drop to ~0.71, demonstrating that standard random K-fold partially overfits to local coordinate clusters.
3. **Chained App Performance:** When users select locations on the map and inputs are estimated via spatial helper models, the app achieves an $R^2$ of **0.83** and MAE of **$31,410**, which is transparently reported in the UI.

All metrics are persisted in [`reports/`](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/reports) and visual figures in [`figures/`](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/figures).

---

## Project Structure

```
├── docs/                        # Complete technical documentation suite
│   ├── README.md                # Documentation navigation hub
│   ├── research_and_dataset.md  # Dataset origin, schema, censoring, and ethics
│   ├── ml_training_and_process.md# Pipelines, OLS math, model tuning, GPU setup
│   ├── results_and_insights.md  # Metrics, t-tests, spatial CV, and residuals
│   └── code_architecture.md     # Module API, app caching, artifact specs
├── housing_features.py          # Shared feature engineering, preprocessors, GPU detection
├── Trainer_helper_model.ipynb   # Helper models for app inference (ocean k-NN, income RF)
├── DMBI_Project_v2.ipynb        # Full exploratory analysis, OLS derivation, model tuning
├── app.py                       # Interactive Folium + Streamlit geospatial web app
├── housing.csv                  # Source census dataset
├── requirements.txt             # Pinned project dependencies
├── figures/                     # 12 analytical plots and decision summary dashboard
├── reports/                     # Output metrics (CSV, JSON, permutation importance)
└── models/                      # Serialized Joblib pipelines and metadata JSON
```

---

## Setup and Installation

### 1. Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name housing-venv
```

**GPU Acceleration:** The installed `xgboost` wheel includes CUDA support. When an NVIDIA GPU (e.g., RTX 3050) and drivers are present, [`housing_features.get_xgb_device()`](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/housing_features.py) selects `"cuda"` automatically; otherwise it gracefully falls back to CPU.

---

## Running the Project

### Retraining Models (Optional)
Pre-trained artifacts are included in [`models/`](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/models). To re-execute training from scratch:
1. Run all cells in [`Trainer_helper_model.ipynb`](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/Trainer_helper_model.ipynb).
2. Run all cells in [`DMBI_Project_v2.ipynb`](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/DMBI_Project_v2.ipynb).

### Launching the Interactive Web Application
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser. Users can select area characteristics in the sidebar, click any location within California on the interactive Folium map, and compute estimated valuations alongside error intervals.

---

## Limitations and Ethical Notice

* **Historical Data (1990):** Dollar values are in 1990 currency and do not reflect current nominal values.
* **Capped Target:** The census capped valuations at $500,001. High-value coastal estates are systematically understated.
* **Algorithmic Redlining:** Geographic coordinates and income correlate with demographic makeup. Valuations must never be used for individual mortgage underwriting, tenant screening, or insurance determination.
