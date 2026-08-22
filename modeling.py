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


bike_df = pd.read_csv("dataset/seoul_bike_daily_data.csv", parse_dates=["date"])

# Data Preprocessing
# =========================================================
## Setting Feature & Target Variables
feature_columns = [
    "district",
    "day_of_week",
    "is_holiday",
    "is_weekend",
    "temp_max",
    "temp_min",
    "feels_like",
    "humidity",
    "precip",
]

target_columns = [
    "use_count",
    "avg_use_time",
]

X = bike_df[feature_columns].copy()
y = bike_df[target_columns].copy()
dates = bike_df["date"].copy()

print("\nX shape:", X.shape)
print("\ny shape:", y.shape)


## Split point for training and testing data (e.g., 80% train, 20% test)
test_start_date = pd.Timestamp("2023-08-01")

categorical_features = [
    "district",
    "day_of_week",
]

numerical_features = [
    "is_holiday",
    "is_weekend",
    "temp_max",
    "temp_min",
    "feels_like",
    "humidity",
    "precip",
]

## Train Test Split
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


## preprocessing for categorical features using OneHotEncoder
categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
)


## Overall Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_transformer,categorical_features),
        ("numerical", "passthrough", numerical_features)
    ],
    remainder="drop"
)

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)



# Modeling
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


## Saving predictions and evaluation results for each model
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
print(evaluation_df)