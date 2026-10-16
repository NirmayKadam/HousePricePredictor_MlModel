# Research and Dataset Specifications

## 1. Domain Background and Origin

The California Housing dataset is a classic benchmark in spatial econometrics and predictive modeling. It was originally compiled by **R. Kelley Pace and Ronald Barry** in their seminal 1997 paper, *"Sparse Spatial Autoregressions"*, published in *Statistics & Probability Letters* (Vol. 33, pp. 291–297). 

The dataset captures socioeconomic and housing characteristics from the **1990 United States Census** for all 20,640 census block groups in California. It was subsequently included in the StatLib archive maintained by Carnegie Mellon University and popularized in Aurélien Géron's *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. Géron's version introduced categorical ocean proximity tags and minor synthetic adjustments (such as 207 missing values in `total_bedrooms`).

---

## 2. Granularity and Unit of Analysis

A common misconception in property price modeling is treating records as individual single-family homes. In this dataset:

* **Unit of Analysis:** One record represents a **Census Block Group**.
* **Population Scale:** A block group is the smallest geographic census division for which the US Census Bureau publishes sample data, typically containing **600 to 3,000 residents** (median population in this dataset is 1,166; mean is 1,425).
* **Aggregate Metrics:** Variables such as `total_rooms` and `total_bedrooms` represent the cumulative sum across all households in the entire block group, rather than attributes of a single property. Per-household rates must be derived via feature engineering.

---

## 3. Schema and Feature Definitions

The raw dataset ([housing.csv](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/housing.csv)) contains **20,640 records** and **10 features**:

| Feature Name | Data Type | Measurement Unit | Description |
|---|---|---|---|
| `longitude` | Continuous Float | Degrees West | East-West coordinate; ranges from -124.35° to -114.31°. |
| `latitude` | Continuous Float | Degrees North | North-South coordinate; ranges from 32.54° to 41.95°. |
| `housing_median_age` | Continuous / Integer | Years | Median age of housing structures in the block group (capped at 52 years). |
| `total_rooms` | Continuous Integer | Count | Aggregated room count across all homes in the block group. |
| `total_bedrooms` | Continuous Float | Count | Aggregated bedroom count; contains 207 missing values (1.003%). |
| `population` | Continuous Integer | Persons | Total resident population in the block group. |
| `households` | Continuous Integer | Count | Total occupied housing units (households) in the block group. |
| `median_income` | Continuous Float | Tens of Thousands (USD) | Median household income in the block group (e.g., 3.8700 represents ~$38,700). |
| `ocean_proximity` | Categorical String | Category | Geographical proximity classification to the Pacific coast. |
| **`median_house_value`** | Continuous Integer | USD | **Target variable**: Median property valuation in the block group (capped at $500,001). |

---

## 4. Ocean Proximity Categories

The categorical variable `ocean_proximity` categorizes the geographic coastal tier:

| Category | Frequency | Percentage | Typical Characteristics |
|---|---|---|---|
| `<1H OCEAN` | 9,136 | 44.26% | Suburban and coastal coastal plain communities within an hour of the coastline. |
| `INLAND` | 6,551 | 31.74% | Central Valley and inland desert regions; lowest average valuations. |
| `NEAR OCEAN` | 2,658 | 12.88% | Directly along the Pacific coast; high valuation premium. |
| `NEAR BAY` | 2,290 | 11.09% | San Francisco Bay Area perimeter; high median incomes and property values. |
| `ISLAND` | 5 | 0.02% | Santa Barbara Channel Islands / Santa Catalina Island; extreme rarity. |

> [!WARNING]
> The `ISLAND` category contains only 5 samples in the entire dataset. In stratified train-test splits, random folds might miss this category entirely, necessitating `handle_unknown="ignore"` within scikit-learn's `OneHotEncoder`.

---

## 5. Data Quality, Anomalies, and Censoring

### 5.1 Missing Value Pattern
* `total_bedrooms` is null for 207 block groups (1.003%).
* The missingness pattern is verified to be Missing Completely at Random (MCAR) or Missing at Random (MAR) across coordinates and income tiers.
* As documented in [ml_training_and_process.md](file:///d:/Nirmay%20pc/ENGINEERING/projects/HousePricePredictor_MlModel/docs/ml_training_and_process.md), median imputation within the cross-validation fold pipeline outperforms row deletion.

### 5.2 Artificially Capped (Censored) Distributions
Exploratory data analysis reveals severe right-censoring in two primary variables:
1. **`median_house_value` (Target):** The 1990 census capped responses at **$500,001**. Approximately 965 blocks (4.68% of the data) are clustered directly at $500,001. A standard regression model trained on this censored data systematically under-predicts ultra-luxury properties.
2. **`housing_median_age`:** Capped at **52 years** (1,273 records or 6.17%). Blocks older than 52 years are compressed into this boundary bin.

### 5.3 Severe Skewness
Distributions of `total_rooms`, `total_bedrooms`, `population`, and `households` exhibit severe positive right-skewness (kurtosis > 10). Linear models require a $\log(1 + x)$ transform on these columns, whereas decision tree ensembles remain invariant to strictly monotonic transformations.

---

## 6. Socioeconomic and Ethical Considerations

### 6.1 Historical Context and Redlining
The data dates to 1990, reflecting historical demographic settlement patterns shaped by 20th-century housing policies, including post-WWII redlining, zoning covenants, and racial steering in California urban centers (e.g., Los Angeles, Oakland, San Francisco).

### 6.2 Proxy Discrimination Risks
* **Geospatial Coordinates as Proxies:** Latitude and longitude, combined with median income, serve as strong proxies for race, ethnicity, and generational wealth.
* **Algorithmic Bias:** Automated valuation models trained purely on historical prices risk perpetuating historical inequities if deployed uncritically for credit underwriting, mortgage approvals, insurance premiums, or municipal tax assessments.
* **Appropriate Domain Use:** This project is intended as an educational benchmark and macroeconomic exploration tool. It is **not suitable for individual lending, appraisal, or rental decisions**.
