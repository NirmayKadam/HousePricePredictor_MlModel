"""
Shared feature engineering and preprocessing for the California house price project.

Both the training notebook and the Streamlit app import from here, so the app always
applies exactly the same transformations the model was trained with.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "median_house_value"
CATEGORICAL = ["ocean_proximity"]
RAW_NUMERIC = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
]
ENGINEERED = ["rooms_per_household", "bedrooms_per_room", "population_per_household"]
NUMERIC = RAW_NUMERIC + ENGINEERED

# Right-skewed columns that get a log1p transform for the linear model
SKEWED = [
    "total_rooms", "total_bedrooms", "population", "households", "median_income",
    "rooms_per_household", "bedrooms_per_room", "population_per_household",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-household ratio features computed from the RAW counts.

    total_bedrooms may be missing; the ratio is then NaN and is imputed later
    inside the pipeline (so imputation is learned from training folds only).
    """
    df = df.copy()
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    return df


def build_preprocessor(kind: str = "tree") -> ColumnTransformer:
    """Column preprocessing.

    kind="linear": median-impute, log1p the skewed columns, standardise everything.
    kind="tree":   median-impute only (trees are invariant to monotonic scaling).
    Both one-hot encode ocean_proximity.
    """
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if kind == "linear":
        # Drop the first category ("<1H OCEAN") to avoid the dummy-variable trap,
        # so the OLS coefficients are identifiable and read as "vs <1H OCEAN".
        onehot = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        other = [c for c in NUMERIC if c not in SKEWED]
        skewed_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ])
        other_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        return ColumnTransformer([
            ("skewed", skewed_pipe, SKEWED),
            ("num", other_pipe, other),
            ("cat", onehot, CATEGORICAL),
        ], verbose_feature_names_out=False)
    if kind == "tree":
        return ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), NUMERIC),
            ("cat", onehot, CATEGORICAL),
        ], verbose_feature_names_out=False)
    raise ValueError(f"unknown kind: {kind!r}")


def make_pipeline(model, kind: str = "tree") -> Pipeline:
    """Full pipeline: raw dataframe in -> prediction out."""
    return Pipeline([
        ("features", FunctionTransformer(add_engineered_features)),
        ("prep", build_preprocessor(kind)),
        ("model", model),
    ])


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """80/20 train-test split, stratified on income bands.

    Income is the strongest predictor, so stratifying keeps its distribution the same
    in both sets. Both notebooks use this so the helper models never see the test rows.
    """
    from sklearn.model_selection import train_test_split
    bands = pd.cut(df["median_income"], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=False)
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=bands)


def get_xgb_device() -> str:
    """Return "cuda" if XGBoost can train on the GPU here, else "cpu"."""
    try:
        import xgboost as xgb
        if not xgb.build_info().get("USE_CUDA", False):
            return "cpu"
        rng = np.random.default_rng(0)
        xgb.XGBRegressor(device="cuda", tree_method="hist", n_estimators=2).fit(
            rng.random((64, 3)), rng.random(64)
        )
        return "cuda"
    except Exception:
        return "cpu"
