# =========================================================
# Seoul Public Bike Demand Forecast - Modeling
# =========================================================


# Importing necessary libraries
# =========================================================
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from pathlib import Path
import joblib

# Loading Dataset
# =========================================================
bike_weather_df= pd.read_csv("dataset/seoul_bike_weather_forecast_data.csv", parse_dates=["date"])

# Defining Feature & Target Variables
# =========================================================
feature_columns = [
    "district",
    "day_of_week",
    "is_holiday",
    "is_weekend",
    "temp_max",
    "temp_min",
    "humidity_mean",
    "precip_prob_max",
    "wind_speed_mean"
]

target_columns = [
    "use_count",
    "avg_use_time",
]

X = bike_weather_df[feature_columns].copy()
y = bike_weather_df[target_columns].copy()
dates = bike_weather_df["date"].copy()

print("\nX shape:", X.shape)
print("\ny shape:", y.shape)


# Train Test Split
# =========================================================
test_start_date = pd.Timestamp("2025-08-01")

train_mask = dates < test_start_date
test_mask = dates >= test_start_date


X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

train_dates = dates.loc[train_mask].copy()
test_dates = dates.loc[test_mask].copy()


## Checking the actual train test split ratios
train_ratio = len(X_train) / len(X)
test_ratio = len(X_test) / len(X)

print(f"\nTrain Ratio: {train_ratio:.2%}")
print(f"\nTest Ratio: {test_ratio:.2%}")


# Preprocessing
# =========================================================
categorical_features = [
    "district",
    "day_of_week",
]

numerical_features = [
    "is_holiday",
    "is_weekend",
    "temp_max",
    "temp_min",
    "humidity_mean",
    "precip_prob_max",
    "wind_speed_mean",
]

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
)

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_transformer,categorical_features),
        ("numerical", "passthrough", numerical_features)
    ],
    remainder="drop"
)

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

processed_feature_names = (preprocessor.get_feature_names_out())


# Modeling
# =========================================================
baseline_train_df = pd.concat(
    [
        X_train[["district"]],
        y_train,
    ],
    axis=1,
)

## Calculate district-level averages
district_baseline = (
    baseline_train_df
    .groupby("district")[target_columns]
    .mean()
)

baseline_results = []
baseline_predictions = {}

for target in target_columns:
    ### For every Test row, use the corresponding district's Train-period average.
    baseline_pred = (
        X_test["district"]
        .map(district_baseline[target])
    )

    baseline_predictions[target] = baseline_pred

    baseline_mae = mean_absolute_error(
        y_test[target],
        baseline_pred,
    )

    baseline_r2 = r2_score(
        y_test[target],
        baseline_pred,
    )

    baseline_results.append({
        "target": target,
        "model": "District Mean Baseline",
        "MAE": baseline_mae,
        "R2": baseline_r2,
    })

baseline_evaluation_df = pd.DataFrame(baseline_results)

print("\n" + "=" * 60)
print("Baseline Evaluation")
print("=" * 60)

print(baseline_evaluation_df.round(3))


# Machine Learning Models
# =========================================================
def create_models():
    models = {
        "Decision Tree": DecisionTreeRegressor(
            max_depth=15,
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            random_state=42
        )
    }
    return models


# Train and Evaluate ML Models
# =========================================================
trained_models = {}
predictions = {}
evaluation_results = []

for target in target_columns:
    # 현재 타깃에 사용할 새로운 모델 3개 생성
    models = create_models()

    trained_models[target] = {}
    predictions[target] = {}

    for model_name, model in models.items():
        model.fit(
            X_train_processed,
            y_train[target]
        )

        y_pred = model.predict(
            X_test_processed
        )

        trained_models[target][model_name] = model
        predictions[target][model_name] = y_pred


        mae = mean_absolute_error(
            y_test[target],
            y_pred,
        )

        r2 = r2_score(
            y_test[target],
            y_pred,
        )

        evaluation_results.append({
            "target": target,
            "model": model_name,
            "MAE": mae,
            "R2": r2,
        })


        print(f"MAE: {mae:.3f}")
        print(f"R² : {r2:.3f}")


evaluation_df = pd.DataFrame(evaluation_results)


# Combining Baseline & ML Results
# =========================================================
all_evaluation_df = pd.concat(
    [baseline_evaluation_df, evaluation_df],
    ignore_index=True,
)


# Calculating Improvement Over Baseline
# =========================================================
baseline_mae_dict = (
    baseline_evaluation_df
    .set_index("target")["MAE"]
    .to_dict()
)


## Calculate how much each model reduced MAE compared with the district mean baseline.
all_evaluation_df["MAE_improvement"] = (
    all_evaluation_df.apply(
        lambda row:
        (
            baseline_mae_dict[row["target"]]
            - row["MAE"]
        )
        / baseline_mae_dict[row["target"]]
        * 100,
        axis=1
    )
)


# Model Evaluation
# =========================================================
print("\n" + "=" * 60)
print("Final Model Evaluation")
print("=" * 60)

print(
    all_evaluation_df.round({
        "MAE": 3,
        "R2": 3,
        "MAE_improvement": 2
    })
)

# Selecting Best Models
# =========================================================
BEST_MODELS = {
    "use_count": "Random Forest",
    "avg_use_time": "HistGradientBoosting",
}


## Creating dataframe of actual test values and predicted values
analysis_df = pd.DataFrame({

    "date": test_dates.reset_index(drop=True),

    "district": X_test["district"].reset_index(drop=True),

    "day_of_week": X_test["day_of_week"].reset_index(drop=True),

    "is_holiday": X_test["is_holiday"].reset_index(drop=True),

    "is_weekend": X_test["is_weekend"].reset_index(drop=True),

    "precip_prob_max": X_test["precip_prob_max"].reset_index(drop=True),

    "use_count_actual": y_test["use_count"].reset_index(drop=True),

    "avg_use_time_actual": y_test["avg_use_time"].reset_index(drop=True),

    "use_count_pred": predictions["use_count"][BEST_MODELS["use_count"]],

    "avg_use_time_pred": predictions["avg_use_time"][BEST_MODELS["avg_use_time"]]
})

## Residual > 0 : predicted value is less than the actual value
## Residual < 0 : predicted value is greater than the actual value
analysis_df["use_count_residual"] = (analysis_df["use_count_actual"] - analysis_df["use_count_pred"])

analysis_df["avg_use_time_residual"] = (analysis_df["avg_use_time_actual"] - analysis_df["avg_use_time_pred"])

## Absolute value of difference
analysis_df["use_count_abs_error"] = (analysis_df["use_count_residual"].abs())

analysis_df["avg_use_time_abs_error"] = (analysis_df["avg_use_time_residual"].abs())

## Actual vs Predicted
### use_count 
plt.figure(figsize=(7, 7))

plt.scatter(
    analysis_df["use_count_actual"],
    analysis_df["use_count_pred"],
    alpha=0.4
)

min_value = min(
    analysis_df["use_count_actual"].min(),
    analysis_df["use_count_pred"].min()
)

max_value = max(
    analysis_df["use_count_actual"].max(),
    analysis_df["use_count_pred"].max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual use_count")
plt.ylabel("Predicted use_count")

plt.title("Random Forest - Actual vs Predicted use_count")

plt.show()

### avg_use_time
plt.figure(figsize=(7, 7))

plt.scatter(
    analysis_df["avg_use_time_actual"],
    analysis_df["avg_use_time_pred"],
    alpha=0.4
)

min_value = min(
    analysis_df["avg_use_time_actual"].min(),
    analysis_df["avg_use_time_pred"].min()
)

max_value = max(
    analysis_df["avg_use_time_actual"].max(),
    analysis_df["avg_use_time_pred"].max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual avg_use_time")
plt.ylabel("Predicted avg_use_time")

plt.title("HistGradientBoosting - Actual vs Predicted avg_use_time")

plt.show()

## District Level Error Analysis
district_error_df = (
    analysis_df
    .groupby("district")
    .agg(

        # Actual use_count for each district
        use_count_actual_mean=(
            "use_count_actual",
            "mean",
        ),

        # use_count MAE for each district
        use_count_MAE=(
            "use_count_abs_error",
            "mean",
        ),

        # avg_use_time for each district
        avg_use_time_actual_mean=(
            "avg_use_time_actual",
            "mean",
        ),

        # avg_use_time MAE for each district
        avg_use_time_MAE=(
            "avg_use_time_abs_error",
            "mean",
        )
    )
    .reset_index()
)


print("\n" + "=" * 60)
print("District Error Analysis")
print("=" * 60)

print(district_error_df.sort_values("use_count_MAE").round(3))

district_use_count_plot = (
    district_error_df
    .sort_values("use_count_MAE")
)


plt.figure(figsize=(10, 8))

plt.barh(
    district_use_count_plot["district"],
    district_use_count_plot["use_count_MAE"],
)

plt.xlabel("MAE")
plt.ylabel("District")

plt.title("Random Forest - use_count MAE by District")

plt.show()

district_avg_time_plot = (district_error_df.sort_values("avg_use_time_MAE"))


plt.figure(figsize=(10, 8))

plt.barh(
    district_avg_time_plot["district"],
    district_avg_time_plot["avg_use_time_MAE"],
)

plt.xlabel("MAE (minutes)")
plt.ylabel("District")

plt.title(
    "HistGradientBoosting - "
    "avg_use_time MAE by District"
)

plt.show()

## Finding district with lowest and highest MAE
print("\nLowest use_count MAE:")
print(
    district_error_df
    .nsmallest(
        5,
        "use_count_MAE",
    )[
        [
            "district",
            "use_count_MAE",
        ]
    ]
)


print("\nHighest use_count MAE:")
print(
    district_error_df
    .nlargest(
        5,
        "use_count_MAE",
    )[
        [
            "district",
            "use_count_MAE",
        ]
    ]
)

print("\nLowest avg_use_time MAE:")
print(
    district_error_df
    .nsmallest(
        5,
        "avg_use_time_MAE",
    )[
        [
            "district",
            "avg_use_time_MAE",
        ]
    ]
)

print("\nHighest avg_use_time MAE:")
print(
    district_error_df
    .nlargest(
        5,
        "avg_use_time_MAE",
    )[
        [
            "district",
            "avg_use_time_MAE",
        ]
    ]
)

## Daily Prediction Trend 
daily_use_count_df = (
    analysis_df
    .groupby("date")
    .agg(
        actual=(
            "use_count_actual",
            "sum",
        ),

        predicted=(
            "use_count_pred",
            "sum",
        )
    )
    .reset_index()
)

daily_use_count_plot = (daily_use_count_df.tail(60))


plt.figure(figsize=(14, 6))

plt.plot(
    daily_use_count_plot["date"],
    daily_use_count_plot["actual"],
    label="Actual"
)

plt.plot(
    daily_use_count_plot["date"],
    daily_use_count_plot["predicted"],
    label="Predicted"
)

plt.xlabel("Date")
plt.ylabel("Total use_count")

plt.title(f"Daily Total use_count - Last {60} Days")

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

daily_avg_time_df = (
    analysis_df
    .groupby("date")
    .agg(
        actual=(
            "avg_use_time_actual",
            "mean",
        ),

        predicted=(
            "avg_use_time_pred",
            "mean",
        )
    )
    .reset_index()
)


daily_avg_time_plot = (daily_avg_time_df.tail(60))


plt.figure(figsize=(14, 6))

plt.plot(
    daily_avg_time_plot["date"],
    daily_avg_time_plot["actual"],
    label="Actual"
)

plt.plot(
    daily_avg_time_plot["date"],
    daily_avg_time_plot["predicted"],
    label="Predicted"
)

plt.xlabel("Date")
plt.ylabel("Average use time")

plt.title(f"Daily Average use time - Last {60} Days")

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

## Performance by Condition
### Rain vs No Rain
analysis_df["rain_condition"] = (
    analysis_df["precip_prob_max"] > 0
)

analysis_df["rain_condition"] = (
    analysis_df["rain_condition"]
    .map({
        True: "Rain",
        False: "No Rain",
    })
)


rain_error_df = (
    analysis_df
    .groupby("rain_condition")
    .agg(

        sample_count=(
            "district",
            "size",
        ),

        use_count_MAE=(
            "use_count_abs_error",
            "mean",
        ),

        avg_use_time_MAE=(
            "avg_use_time_abs_error",
            "mean",
        ),
    )
)


print("\n" + "=" * 60)
print("Rain vs No Rain")
print("=" * 60)

print(
    rain_error_df.round(3)
)

### Weekday vs Weekend
analysis_df["weekend_condition"] = (
    analysis_df["is_weekend"]
    .map({
        True: "Weekend",
        False: "Weekday"
    })
)

weekend_error_df = (
    analysis_df
    .groupby("weekend_condition")
    .agg(

        sample_count=(
            "district",
            "size"
        ),

        use_count_MAE=(
            "use_count_abs_error",
            "mean"
        ),

        avg_use_time_MAE=(
            "avg_use_time_abs_error",
            "mean"
        )
    )
)


print("\n" + "=" * 60)
print("Weekday vs Weekend")
print("=" * 60)

print(
    weekend_error_df.round(3)
)

## Feature Importance
### Random Forest Feature Importance
use_count_model = (
    trained_models["use_count"][
        BEST_MODELS["use_count"]
    ]
)


use_count_importance_df = pd.DataFrame({
    "feature": processed_feature_names,
    "importance": (use_count_model.feature_importances_)
})


use_count_importance_df = use_count_importance_df.sort_values("importance", ascending=False)


print("\n" + "=" * 60)
print("use_count Feature Importance")
print("=" * 60)

print(
    use_count_importance_df
    .head(15)
    .round(4)
)

top_use_count_features = (
    use_count_importance_df
    .head(15)
    .sort_values("importance")
)


plt.figure(figsize=(10, 7))

plt.barh(
    top_use_count_features["feature"],
    top_use_count_features["importance"]
)

plt.xlabel("Feature Importance")

plt.title("Random Forest - Feature Importance for use_count")

plt.tight_layout()

plt.show()


# Saving Models 
# =========================================================
MODEL_DIR = Path("models")

## Create folder 
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

## Model for use_count
final_use_count_model = trained_models["use_count"]["Random Forest"]

## Model for avg_use_time
final_avg_use_time_model = trained_models["avg_use_time"]["HistGradientBoosting"]


## Saving preprocessor & final models
joblib.dump(
    preprocessor,
    MODEL_DIR / "preprocessor.pkl",
    compress = 3
)

joblib.dump(
    final_use_count_model,
    MODEL_DIR / "use_count_model.pkl",
    compress = 3
)

joblib.dump(
    final_avg_use_time_model,
    MODEL_DIR / "avg_use_time_model.pkl",
    compress = 3
)

district_baseline.to_csv(
    MODEL_DIR / "district_baseline.csv",
    encoding="utf-8-sig"
)

## Saving model information
model_info = {
    "feature_columns": feature_columns,

    "categorical_features": categorical_features,

    "numerical_features": numerical_features,

    "use_count_model": "Random Forest",

    "avg_use_time_model": "HistGradientBoosting",

    "test_start_date": "2025-08-01"
}


joblib.dump(
    model_info,
    MODEL_DIR / "model_info.pkl",
    compress = 3
)