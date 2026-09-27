# Seoul Public Bike Demand Forecast

An end-to-end machine learning project for forecasting **next-day Seoul Public Bike (따릉이) usage across Seoul's 25 districts**.

The project combines historical Seoul Public Bike usage data with Korea Meteorological Administration (KMA) weather forecasts, builds district-level daily features, compares multiple regression models, performs time-aware evaluation, and serves live predictions through an interactive Streamlit dashboard.

The system predicts:

- **Daily rental count** for each Seoul district
- **Average rental duration** for each Seoul district

A key design choice is that the model uses the **KMA weather forecast available at 20:00 KST on the previous day**, rather than realized future weather. This keeps the forecasting setup aligned with information that would actually be available at prediction time.

---

## Live Dashboard

**Live app:**  
`[ADD YOUR STREAMLIT URL HERE]`

The deployed dashboard provides:

- an interactive choropleth map of Seoul's 25 districts,
- predicted bike rental demand,
- predicted average rental duration,
- hover-based district details,
- Top 3 districts for the selected metric,
- forecast date and model update time,
- automatic switching between today's and tomorrow's forecast based on the 20:00 KST KMA forecast release.

### Dashboard Preview

Add a screenshot of the deployed dashboard to the repository, for example:

```text
README_assets/
└── dashboard_preview.png
```

Then embed it here:

```markdown
![Dashboard Preview](README_assets/dashboard_preview.png)
```

---

## 1. Project Motivation

Seoul Public Bike, commonly known as **따릉이**, is Seoul's public bicycle-sharing system.

Bike usage changes substantially depending on factors such as:

- weather,
- weekday/weekend patterns,
- holidays,
- district,
- and seasonal conditions.

The goal of this project is to build a forecasting pipeline that answers:

> **How many Seoul Public Bike rentals are expected for the next prediction date in each Seoul district, and how long are those rentals expected to last on average?**

Instead of predicting only a citywide total, the system forecasts separately for all **25 Seoul autonomous districts (구)**.

This geographic granularity preserves district-level differences and makes the output suitable for geographic visualization.

---

## 2. Prediction Targets

The project predicts two targets.

### 2.1 Daily Rental Count

`use_count`

The total number of Seoul Public Bike rentals within a district on a given day.

This is treated as the primary **demand prediction** target.

### 2.2 Average Use Time

`avg_use_time`

The average rental duration for bikes used within a district on a given day.

The final model therefore produces two predictions for every district:

```text
district
├── predicted_use_count
└── predicted_avg_use_time
```

---

## 3. Prediction Granularity

Each modeling observation represents:

```text
1 date × 1 Seoul district
```

Since Seoul contains 25 autonomous districts, a complete prediction date contains:

```text
25 rows
```

Example inference output:

```text
prediction_date    district    predicted_use_count    predicted_avg_use_time
2026-09-26         강남구      ...                    ...
2026-09-26         강동구      ...                    ...
2026-09-26         강북구      ...                    ...
...                ...         ...                    ...
```

Station-level bike usage is aggregated to the district level before modeling.

---

## 4. Forecasting Setup

One of the most important parts of this project is the distinction between:

- **actual future weather**, and
- **weather forecasts available before the target date**.

Using realized weather for a next-day prediction would introduce information that would not have been known at forecasting time.

To avoid this, the model uses KMA forecast data issued at:

```text
20:00 KST on the previous day
```

for the relevant prediction date.

The application follows this rule:

```text
Before 20:00 KST
─────────────────────────────────
Yesterday's 20:00 KMA forecast
                ↓
Today's bike usage prediction


At or after 20:00 KST
─────────────────────────────────
Today's 20:00 KMA forecast
                ↓
Tomorrow's bike usage prediction
```

For example:

```text
Current time:
2026-09-25 19:30 KST

Weather forecast used:
2026-09-24 20:00 KST

Prediction target:
2026-09-25
```

After the 20:00 forecast becomes available:

```text
Current time:
2026-09-25 20:30 KST

Weather forecast used:
2026-09-25 20:00 KST

Prediction target:
2026-09-26
```

This switching rule is implemented in `get_prediction_context()` inside `inference.py`.

---

## 5. Data Sources

### 5.1 Seoul Public Bike Data

Bike usage and station-related data are collected from the **Seoul Open Data Plaza**.

The bike data pipeline converts station-level information into daily district-level observations.

Conceptually:

```text
Seoul Public Bike station data
                +
daily usage data
                ↓
station ID mapping
                ↓
station → district
                ↓
aggregate by date + district
                ↓
use_count
avg_use_time
```

Relevant data includes:

- Seoul Public Bike station information
- Seoul Public Bike usage information

---

### 5.2 Korea Meteorological Administration Forecast Data

Weather forecasts are retrieved from the **Korea Meteorological Administration API Hub**.

The weather pipeline retrieves nationwide KMA forecast grids and extracts the grid value corresponding to each Seoul district.

The project uses the following weather variables:

| KMA Variable | Modeling Feature | Description |
|---|---|---|
| `TMX` | `temp_max` | Daily maximum temperature |
| `TMN` | `temp_min` | Daily minimum temperature |
| `REH` | `humidity_mean` | Mean relative humidity |
| `POP` | `precip_prob_max` | Maximum precipitation probability |
| `WSD` | `wind_speed_mean` | Mean wind speed |

Hourly forecasts are aggregated into daily district-level features.

Examples:

```text
REH hourly forecasts
        ↓
mean
        ↓
humidity_mean
```

```text
POP hourly forecasts
        ↓
maximum
        ↓
precip_prob_max
```

```text
WSD hourly forecasts
        ↓
mean
        ↓
wind_speed_mean
```

---

### 5.3 Seoul District to KMA Grid Mapping

KMA weather forecasts are provided on a national grid.

Each Seoul district is mapped to a corresponding KMA grid coordinate:

```text
district
├── nx
└── ny
```

The mapping is stored in:

```text
dataset/seoul_district_kma_grid.csv
```

This allows the weather pipeline to retrieve one forecast value for each of Seoul's 25 districts from the nationwide KMA grid.

---

### 5.4 Seoul District Boundary Data

The dashboard uses a GeoJSON file containing Seoul's district boundaries.

Source:

- `cubensys/Korea_District`
- https://github.com/cubensys/Korea_District

The file used by the project is stored as:

```text
dataset/seoul_districts.geojson
```

Each GeoJSON feature contains district information such as:

```text
SIG_CD
SIG_ENG_NM
SIG_KOR_NM
```

The dashboard joins model predictions to the GeoJSON using the Korean district name:

```text
prediction["district"]
        ↕
GeoJSON["properties"]["SIG_KOR_NM"]
```

---

## 6. Dataset Construction

The project builds the final modeling dataset by combining bike usage and weather forecasts at the same geographic and temporal level.

The final join key is:

```text
date × district
```

The general pipeline is:

```text
Seoul Public Bike API
        ↓
station-level usage
        ↓
district-level aggregation
        ↓
daily bike dataset
                         \
                          \
                           → merge by date + district
                          /
                         /
KMA Forecast API
        ↓
national weather grids
        ↓
Seoul district extraction
        ↓
daily weather features
```

The resulting modeling dataset includes fields such as:

```text
date
district
day_of_week
is_holiday
is_weekend
temp_max
temp_min
humidity_mean
precip_prob_max
wind_speed_mean
use_count
avg_use_time
```

---

## 7. Features

The current model uses nine input features.

### Spatial Feature

```text
district
```

### Calendar Features

```text
day_of_week
is_holiday
is_weekend
```

### Weather Features

```text
temp_max
temp_min
humidity_mean
precip_prob_max
wind_speed_mean
```

The feature list used for training is:

```python
feature_columns = [
    "district",
    "day_of_week",
    "is_holiday",
    "is_weekend",
    "temp_max",
    "temp_min",
    "humidity_mean",
    "precip_prob_max",
    "wind_speed_mean",
]
```

The categorical variables are:

```text
district
day_of_week
```

and are encoded using the preprocessing pipeline saved during model training.

The remaining variables are passed through as numerical features.

---

## 8. Train/Test Strategy

Because this is a forecasting problem, the dataset is **not randomly shuffled** before evaluation.

Instead, the project uses a chronological split.

The test period begins on:

```text
2025-08-01
```

Therefore:

```text
Dates before 2025-08-01
        ↓
Training set

Dates on or after 2025-08-01
        ↓
Test set
```

This ensures that evaluation is performed on observations occurring after the training period.

---

## 9. Baseline Model

Before comparing machine learning algorithms, the project defines a simple district-level historical mean baseline.

For each Seoul district, the training-period mean is calculated separately for:

```text
use_count
avg_use_time
```

For every test observation, the corresponding district's historical training mean becomes the baseline prediction.

Conceptually:

```text
Training data
        ↓
mean use_count by district
mean avg_use_time by district
        ↓
baseline test predictions
```

This gives a simple reference point for answering:

> Does the machine learning model outperform using a district's historical average?

The project reports MAE improvement relative to this baseline.

---

## 10. Machine Learning Models

Three tree-based regression models are compared for each target.

### Decision Tree

```python
DecisionTreeRegressor(
    max_depth=15,
    random_state=42,
)
```

### Random Forest

```python
RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
```

### Histogram Gradient Boosting

```python
HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    random_state=42,
)
```

The models are trained independently for:

- `use_count`
- `avg_use_time`

---

## 11. Model Selection

The selected models are:

| Target | Selected Model |
|---|---|
| `use_count` | Random Forest |
| `avg_use_time` | HistGradientBoosting |

These selected models are serialized and reused by the live inference pipeline.

---

## 12. Model Performance

Evaluation uses:

- Mean Absolute Error (MAE)
- R²
- MAE improvement over the district-mean baseline

### 12.1 Daily Rental Count

| Model | MAE | R² | MAE Improvement vs Baseline |
|---|---:|---:|---:|
| District Mean Baseline | 1331.23 | 0.681 | 0.00% |
| Decision Tree | 1010.12 | 0.795 | 24.12% |
| **Random Forest** | **628.48** | **0.919** | **52.79%** |
| HistGradientBoosting | 642.17 | 0.918 | 51.76% |

The selected Random Forest model reduces MAE by approximately:

```text
52.79%
```

relative to the district-level historical mean baseline.

### 12.2 Average Use Time

| Model | MAE | R² | MAE Improvement vs Baseline |
|---|---:|---:|---:|
| District Mean Baseline | 2.333 min | 0.371 | 0.00% |
| Decision Tree | 1.514 min | 0.715 | 35.12% |
| Random Forest | 1.237 min | 0.794 | 46.96% |
| **HistGradientBoosting** | **1.165 min** | **0.819** | **50.06%** |

The selected HistGradientBoosting model reduces MAE by approximately:

```text
50.06%
```

relative to the district-level historical mean baseline.

---

## 13. Model Evaluation and Error Analysis

The modeling pipeline includes more than global MAE and R².

It also evaluates model behavior through:

- actual vs. predicted plots,
- district-level MAE,
- daily predicted vs. actual trends,
- weekday vs. weekend performance,
- feature importance,
- permutation importance.

Generated figures are saved under:

```text
results/figures/
```

Examples include:

```text
use_count_actual_vs_predicted.png
avg_use_time_actual_vs_predicted.png
use_count_mae_by_district.png
avg_use_time_mae_by_district.png
daily_use_count_last_60_days.png
daily_avg_use_time_last_60_days.png
use_count_feature_importance.png
avg_use_time_permutation_importance.png
```

The modeling script also saves test predictions for later inference validation:

```text
results/modeling_test_predictions.csv
```

---

## 14. Feature Importance

Different interpretation methods are used for the two selected models.

### Rental Count

The Random Forest model provides native tree-based feature importance.

The project extracts and visualizes feature importance for the processed feature space.

### Average Use Time

`HistGradientBoostingRegressor` does not expose the same native feature-importance interface.

Therefore, the project uses **permutation importance**.

Conceptually:

```text
original model error
        ↓
shuffle one feature
        ↓
recompute error
        ↓
measure performance degradation
```

A larger degradation indicates that the model depends more strongly on that feature.

---

## 15. Saved Model Artifacts

After model selection, the following artifacts are saved:

```text
models/
├── preprocessor.pkl
├── use_count_model.pkl
├── avg_use_time_model.pkl
├── model_info.pkl
└── model_evaluation.csv
```

These artifacts allow inference to reuse exactly the same preprocessing transformations and trained models without retraining whenever a forecast is requested.

---

## 16. Inference Pipeline

`inference.py` converts live weather forecasts into district-level model predictions.

The inference pipeline performs the following steps:

```text
Current KST time
        ↓
Determine prediction context
        ↓
Select target date
        ↓
Select correct KMA 20:00 forecast
        ↓
Retrieve district-level weather features
        ↓
Validate all 25 districts
        ↓
Generate calendar features
        ↓
Apply saved preprocessing pipeline
        ↓
Run saved models
        ↓
Return 25 district predictions
```

The inference output contains fields such as:

```text
prediction_date
forecast_issued_at
district
predicted_use_count
predicted_avg_use_time
```

---

## 17. Inference Validation

A historical smoke test is included to verify that the saved model artifacts reproduce predictions originally generated during modeling.

The smoke test:

1. selects a historical test date,
2. reconstructs the corresponding weather features,
3. loads the saved preprocessing and model artifacts,
4. generates inference predictions,
5. compares them against `modeling.py` predictions,
6. checks numerical equality with `numpy.allclose()`.

This helps detect inconsistencies caused by:

- preprocessing changes,
- model serialization,
- feature ordering,
- inference implementation differences.

---

## 18. Live Weather Retrieval

`weather_data_gathering.py` handles KMA forecast retrieval.

For each weather variable, nationwide forecast grids are requested from the KMA API.

The system:

```text
KMA nationwide grid
        ↓
district grid coordinates
        ↓
extract 25 Seoul values
        ↓
aggregate hourly forecasts
        ↓
district-level daily weather features
```

To reduce repeated API traffic, downloaded grid arrays are cached locally:

```text
kma_forecast_cache/
```

The cache is organized by forecast issue date and forecast variable.

---

## 19. Interactive Dashboard

The project includes an interactive dashboard built with:

- Streamlit
- Plotly
- GeoJSON

The dashboard directly calls the live inference pipeline.

The general flow is:

```text
dashboard.py
        ↓
get_prediction_context()
        ↓
run_live_inference()
        ↓
get_weather_for_inference()
        ↓
KMA API
        ↓
saved ML models
        ↓
25 district predictions
        ↓
interactive Seoul map
```

---

## 20. Dashboard Features

### 20.1 Choropleth Map

Districts are colored according to the selected prediction metric.

A green continuous color scale is used to visually align the interface with Seoul Public Bike branding.

District boundaries are outlined to improve geographic separation.

### 20.2 Metric Selector

Users can switch between:

```text
Rental Demand
Average Use Time
```

The selected metric determines:

- map color,
- legend,
- Top 3 ranking.

### 20.3 Hover Information

Moving the cursor over any Seoul district displays:

```text
District name
Predicted rentals
Predicted average use time
```

No additional click is required.

### 20.4 Top 3 Districts

The side panel dynamically shows the three highest districts for the selected metric.

For Rental Demand:

```text
Highest predicted rental counts
```

For Average Use Time:

```text
Longest predicted average rental duration
```

### 20.5 Forecast Metadata

The dashboard displays:

```text
Forecast for
Last updated
```

It also explains whether the user is viewing:

- today's latest available prediction, or
- tomorrow's newly generated prediction.

### 20.6 20:00 KST Behavior

Before 20:00 KST:

```text
Today's latest prediction is displayed.
Tomorrow's prediction is not yet available.
```

At or after 20:00 KST:

```text
The dashboard switches to tomorrow's prediction
using the KMA forecast issued at 20:00 KST.
```

---

## 21. Deployment

The dashboard is deployed using **Streamlit Community Cloud**.

Deployment configuration includes:

```text
Entrypoint:
dashboard.py

Python:
3.12
```

The deployment environment uses pinned machine-learning package versions to keep serialized scikit-learn models compatible with the environment in which they were created.

Important package versions include:

```text
scikit-learn==1.8.0
joblib==1.5.3
numpy==2.0.0
pandas==2.3.0
```

---

## 22. API Key Management

The KMA API key is not committed to GitHub.

For local execution, it is read from:

```text
KMA_API_KEY
```

For Streamlit Community Cloud, the key is stored through the deployment's **Secrets** configuration.

Example:

```toml
KMA_API_KEY = "YOUR_KEY"
```

Secret files such as:

```text
.env
.streamlit/secrets.toml
```

should remain excluded from version control.

---

## 23. Project Structure

```text
Seoul_Public_Bike_Demand_Forecast/
│
├── dashboard.py
│
├── inference.py
├── modeling.py
├── preprocessing.py
│
├── bike_data_gathering.py
├── weather_data_gathering.py
├── kma_mapping.py
│
├── dataset/
│   ├── seoul_bike_weather_forecast_data.csv
│   ├── seoul_district_kma_grid.csv
│   ├── seoul_districts.geojson
│   ├── daily_weather_data/
│   └── ...
│
├── models/
│   ├── preprocessor.pkl
│   ├── use_count_model.pkl
│   ├── avg_use_time_model.pkl
│   ├── model_info.pkl
│   └── model_evaluation.csv
│
├── results/
│   ├── modeling_test_predictions.csv
│   └── figures/
│       ├── use_count_actual_vs_predicted.png
│       ├── avg_use_time_actual_vs_predicted.png
│       ├── use_count_mae_by_district.png
│       ├── avg_use_time_mae_by_district.png
│       ├── daily_use_count_last_60_days.png
│       ├── daily_avg_use_time_last_60_days.png
│       ├── use_count_feature_importance.png
│       └── avg_use_time_permutation_importance.png
│
├── kma_forecast_cache/
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
├── .gitignore
└── README.md
```

Some generated datasets, caches, and secret files may be excluded through `.gitignore`.

---

## 24. Running the Project Locally

### Clone the Repository

```bash
git clone https://github.com/jaeho0726/Seoul_Public_Bike_Demand_Forecast.git
cd Seoul_Public_Bike_Demand_Forecast
```

### Install Dependencies

Using Python 3.12:

```bash
python3.12 -m pip install -r requirements.txt
```

### Set the KMA API Key

macOS / Linux:

```bash
export KMA_API_KEY="YOUR_KMA_API_KEY"
```

Confirm that the environment variable is available:

```bash
echo $KMA_API_KEY
```

### Launch the Dashboard

```bash
streamlit run dashboard.py
```

The local dashboard should open at:

```text
http://localhost:8501
```

---

## 25. Reproducing Model Training

To rebuild the models from the processed modeling dataset:

```bash
python modeling.py
```

The script:

1. loads the merged bike-weather dataset,
2. creates the chronological train/test split,
3. preprocesses categorical variables,
4. creates the district-level baseline,
5. trains three machine learning models for each target,
6. evaluates each model,
7. generates diagnostic figures,
8. saves selected models and metadata.

The selected artifacts are written to:

```text
models/
```

and evaluation outputs are written to:

```text
results/
```

---

## 26. Technology Stack

### Programming

- Python 3.12

### Data Processing

- pandas
- NumPy

### Machine Learning

- scikit-learn
- joblib

### API / Data Collection

- Requests
- Seoul Open Data Plaza
- Korea Meteorological Administration API Hub

### Visualization

- Matplotlib
- Plotly
- GeoJSON

### Application

- Streamlit

### Deployment

- Streamlit Community Cloud

---

## 27. Current Limitations

### District-Level Resolution

Predictions are generated for Seoul's 25 districts.

The model does not currently forecast:

- individual bike stations,
- neighborhoods,
- hourly demand.

District aggregation improves stability and visualization simplicity, but removes station-level variation.

### No Historical Demand Lag Features

The current feature set does not include demand-history features such as:

```text
previous-day rentals
previous-week rentals
7-day rolling demand
district rolling averages
```

The model therefore primarily learns demand from:

```text
district
calendar information
weather forecast
```

rather than recent demand trajectories.

### Forecast Completeness

Hourly KMA weather requests may occasionally fail.

The weather pipeline currently aggregates available observations when some hourly values are unavailable.

Additional completeness thresholds could be added before allowing inference.

### GeoJSON Source

The current Seoul district boundary file is based on a third-party GitHub dataset.

A future version could replace this with a directly retrieved current government administrative-boundary dataset.

### Model Validation

The current evaluation uses one chronological holdout period.

A stronger future evaluation could use:

```text
rolling validation
walk-forward validation
multiple temporal test windows
```

### Operational Demand vs. Shortage Risk

A district with high predicted rental demand is not necessarily experiencing a shortage.

The project does not currently include:

- number of available bikes,
- station capacity,
- dock availability,
- redistribution operations,
- maintenance constraints.

Therefore, the dashboard should be interpreted as a **demand forecast**, not a shortage-risk or rebalancing recommendation system.

---

## 28. Future Improvements

### Modeling

- Add lagged demand features
- Add rolling demand statistics
- Perform systematic hyperparameter tuning
- Compare additional boosting methods
- Test time-series-specific models
- Add walk-forward validation

### Spatial Resolution

- Move from district-level to station-level prediction
- Model neighborhood-level variation
- Incorporate station density

### Data

- Add bike availability data
- Add station capacity
- Add precipitation amount
- Add snowfall
- Add air-quality information
- Add public-event information
- Add transportation accessibility variables

### Forecasting System

- Persist daily predictions
- Compare forecasts with actual values over time
- Add model-performance monitoring
- Detect data drift
- Detect forecast-data failures

### Dashboard

- Add historical forecast views
- Add model-error visualization
- Add trend charts by district
- Add downloadable prediction tables
- Add comparison between multiple districts

### Operations

A future version could combine predicted demand with:

```text
bike availability
station capacity
redistribution cost
```

to estimate:

```text
shortage risk
surplus risk
bike rebalancing priority
```

---

## 29. Key Takeaways

This project demonstrates an end-to-end data science workflow:

```text
public data collection
        ↓
data validation
        ↓
spatial mapping
        ↓
feature engineering
        ↓
time-aware machine learning evaluation
        ↓
model serialization
        ↓
live API inference
        ↓
interactive geographic visualization
        ↓
cloud deployment
```

The project focuses not only on predictive performance, but also on making the forecasting process realistic by ensuring that weather inputs correspond to information that would actually have been available at prediction time.

---

## 30. Results Summary

### Daily Rental Count

```text
Model: Random Forest
MAE: 628.48 rentals
R²: 0.919
MAE improvement over district baseline: 52.79%
```

### Average Use Time

```text
Model: HistGradientBoosting
MAE: 1.165 minutes
R²: 0.819
MAE improvement over district baseline: 50.06%
```

These results indicate that district, calendar, and forecast-weather features provide substantial predictive value beyond using each district's historical mean alone.

---

## 31. Links

### Repository

https://github.com/jaeho0726/Seoul_Public_Bike_Demand_Forecast

### Live Dashboard

`[ADD YOUR STREAMLIT URL HERE]`

### Seoul District GeoJSON Source

https://github.com/cubensys/Korea_District

---

## Author

**Jaeho Shim**
