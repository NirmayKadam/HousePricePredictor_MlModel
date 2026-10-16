# Project Documentation Hub

Welcome to the technical and scientific documentation for the **California House Price Predictor** project.

## Documentation Index

1. [Research and Dataset Specifications](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/docs/research_and_dataset.md)
   * Domain background, academic origins (Pace & Barry 1997), and StatLib census compilation.
   * Granularity analysis: Census block groups vs individual properties.
   * Feature schema, distributions, and right-censoring at $500,001.
   * Ethical implications: Redlining, location proxies, and algorithmic fairness caveats.

2. [Machine Learning Training and Engineering Process](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/docs/ml_training_and_process.md)
   * End-to-end scikit-learn Pipeline architecture and transformation rules.
   * Stratified 80/20 train-test splitting on income tiers.
   * Outlier analysis (IQR vs Isolation Forest) and missing value strategies.
   * Engineered per-household ratio features.
   * First-principles derivation of Ordinary Least Squares (Normal Equations, SVD `lstsq`, statsmodels OLS).
   * Model zoo training: Baseline, Linear Regression, Random Forest (CPU), and XGBoost (CUDA GPU).
   * RandomizedSearchCV hyperparameter optimization configurations.
   * Spatial helper models (k-NN ocean proximity classifier & Random Forest median income regressor).

3. [Experimental Results and Key Insights](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/docs/results_and_insights.md)
   * Comprehensive cross-validation vs held-out test benchmarks.
   * Paired two-tailed t-test confirming statistical significance ($p < 0.001$).
   * Spatial autocorrelation study: 0.5° coordinate holdout testing via `Spatial GroupKFold`.
   * Residual diagnostics: Prediction scatter plots, normality Q-Q analysis, and geographic error maps.
   * Predictor interpretability: OLS coefficients, Gini importance, and Permutation Importance rankings.
   * Chained Streamlit app real-world error measurement.

4. [Code Architecture and Technical Reference](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/docs/code_architecture.md)
   * System design and project layout.
   * Function-level documentation for [housing_features.py](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/housing_features.py).
   * Streamlit application internals, geospatial coordinate filtering, and caching in [app.py](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/app.py).
   * Saved artifact specifications and JSON schemas.
   * Developer workflows, environment setup, and execution commands.
