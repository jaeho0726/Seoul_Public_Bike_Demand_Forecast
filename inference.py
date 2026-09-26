# ============================================================
# Seoul Public Bike Demand Forecast - Inference
# ============================================================
import joblib
import numpy as np
import pandas as pd

from datetime import datetime, timedelta, time
from pathlib import Path
from zoneinfo import ZoneInfo

from holidayskr import is_holiday



# Configuration
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"

KST = ZoneInfo("Asia/Seoul")

EXPECTED_DISTRICT_COUNT = 25

WEATHER_COLUMNS = [
    "district",
    "temp_max",
    "temp_min",
    "humidity_mean",
    "precip_prob_max",
    "wind_speed_mean",
]



# Loading Model Artifacts
# ============================================================
preprocessor = joblib.load(
    MODEL_DIR / "preprocessor.pkl"
)

use_count_model = joblib.load(
    MODEL_DIR / "use_count_model.pkl"
)

avg_use_time_model = joblib.load(
    MODEL_DIR / "avg_use_time_model.pkl"
)

model_info = joblib.load(
    MODEL_DIR / "model_info.pkl"
)



# Prediction Time Rule
# ============================================================
def get_prediction_context(now=None):
    """
    Determine which date should be predicted based on
    the daily 20:00 KST KMA forecast switching rule.

    Before 20:00 KST:
        Yesterday 20:00 forecast -> Today prediction

    At or after 20:00 KST:
        Today 20:00 forecast -> Tomorrow prediction
    """

    if now is None:
        now = datetime.now(KST)

    elif now.tzinfo is None:
        now = now.replace(tzinfo=KST)

    else:
        now = now.astimezone(KST)

    if now.hour >= 20:
        forecast_issued_at = now.replace(
            hour=20,
            minute=0,
            second=0,
            microsecond=0,
        )

        prediction_date = (
            now + timedelta(days=1)
        ).date()

    else:
        forecast_issued_at = (
            now - timedelta(days=1)
        ).replace(
            hour=20,
            minute=0,
            second=0,
            microsecond=0,
        )

        prediction_date = now.date()

    return {
        "current_time": now,
        "prediction_date": prediction_date,
        "forecast_issued_at": forecast_issued_at,
        "tmfc": forecast_issued_at.strftime("%Y%m%d%H"),
        "target_date": prediction_date.strftime("%Y%m%d"),
    }



# Input Validation
# ============================================================
def validate_weather_data(weather_df):
    """
    Validate district-level KMA weather features before
    sending them into the ML models.
    """

    missing_columns = [
        column
        for column in WEATHER_COLUMNS
        if column not in weather_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing weather columns: {missing_columns}"
        )

    if len(weather_df) != EXPECTED_DISTRICT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_DISTRICT_COUNT} rows, "
            f"but received {len(weather_df)} rows."
        )

    if (
        weather_df["district"].nunique()
        != EXPECTED_DISTRICT_COUNT
    ):
        raise ValueError(
            "Weather data must contain exactly "
            "25 unique Seoul districts."
        )

    if weather_df[WEATHER_COLUMNS].isnull().any().any():
        null_columns = (
            weather_df[WEATHER_COLUMNS]
            .isnull()
            .sum()
        )

        null_columns = null_columns[
            null_columns > 0
        ]

        raise ValueError(
            f"Null values found in weather data:\n"
            f"{null_columns}"
        )

    if (
        weather_df["temp_max"]
        < weather_df["temp_min"]
    ).any():
        raise ValueError(
            "temp_max cannot be lower than temp_min."
        )

    if not weather_df["humidity_mean"].between(
        0, 100
    ).all():
        raise ValueError(
            "humidity_mean must be between 0 and 100."
        )

    if not weather_df["precip_prob_max"].between(
        0, 100
    ).all():
        raise ValueError(
            "precip_prob_max must be between 0 and 100."
        )

    if (
        weather_df["wind_speed_mean"] < 0
    ).any():
        raise ValueError(
            "wind_speed_mean cannot be negative."
        )



# Preparing Model Input
# ============================================================
def prepare_model_input(
    weather_df,
    prediction_date,
):
    """
    Convert district-level weather data into the same
    feature structure used during model training.
    """

    weather_df = weather_df.copy()

    weather_df["district"] = (
        weather_df["district"]
        .astype(str)
        .str.strip()
    )

    validate_weather_data(weather_df)

    trained_districts = set(
        preprocessor
        .named_transformers_["categorical"]
        .categories_[0]
    )

    input_districts = set(
        weather_df["district"]
    )

    if input_districts != trained_districts:
        missing_districts = (
            trained_districts
            - input_districts
        )

        unknown_districts = (
            input_districts
            - trained_districts
        )

        raise ValueError(
            "District mismatch detected.\n"
            f"Missing districts: "
            f"{sorted(missing_districts)}\n"
            f"Unknown districts: "
            f"{sorted(unknown_districts)}"
        )

    weather_df = (
        weather_df
        .sort_values("district")
        .reset_index(drop=True)
    )

    prediction_timestamp = pd.Timestamp(
        prediction_date
    )

    weather_df["day_of_week"] = (
        prediction_timestamp.day_name()
    )

    weather_df["is_weekend"] = (
        prediction_timestamp.weekday() >= 5
    )

    weather_df["is_holiday"] = bool(
        is_holiday(
            prediction_timestamp.strftime(
                "%Y-%m-%d"
            )
        )
    )

    feature_columns = (
        model_info["feature_columns"]
    )

    model_input_df = (
        weather_df[
            feature_columns
        ]
        .copy()
    )

    return weather_df, model_input_df



# Bike Demand Prediction
# ============================================================
def predict_bike_demand(
    weather_df,
    context=None,
):
    """
    Predict district-level bike demand.

    Parameters
    ----------
    weather_df:
        DataFrame containing weather forecast features
        for all 25 Seoul districts.

    context:
        Prediction context generated by
        get_prediction_context(). If None, current KST
        is used.

    Returns
    -------
    prediction_df:
        DataFrame containing weather information and
        predicted bike demand for each district.

    context:
        Prediction metadata including prediction date,
        KMA forecast issue time, and inference generation time.
    """

    if context is None:
        context = get_prediction_context()

    prediction_date = (
        context["prediction_date"]
    )

    weather_df, model_input_df = (
        prepare_model_input(
            weather_df,
            prediction_date,
        )
    )

    processed_input = (
        preprocessor.transform(
            model_input_df
        )
    )

    predicted_use_count = (
        use_count_model.predict(
            processed_input
        )
    )

    predicted_avg_use_time = (
        avg_use_time_model.predict(
            processed_input
        )
    )

    prediction_df = weather_df.copy()

    prediction_df.insert(
        0,
        "prediction_date",
        pd.Timestamp(
            prediction_date
        ),
    )

    prediction_df.insert(
        1,
        "forecast_issued_at",
        context["forecast_issued_at"],
    )

    prediction_df[
        "predicted_use_count"
    ] = predicted_use_count

    prediction_df[
        "predicted_avg_use_time"
    ] = predicted_avg_use_time

    context = context.copy()
    context["generated_at"] = datetime.now(KST)

    return prediction_df, context


# Complete Live Inference Pipeline
# ============================================================
def run_live_inference(context=None):
    """
    Run the complete live inference pipeline:

    1. Determine the correct prediction date and KMA forecast.
    2. Retrieve district-level KMA weather features.
    3. Run the saved ML models.
    4. Return 25 district predictions and metadata.
    """

    if context is None:
        context = get_prediction_context()

    # Local import keeps the model/inference module independent
    # until live weather retrieval is actually requested.
    from weather_data_gathering import (
        get_weather_for_inference,
    )

    weather_df = get_weather_for_inference(
        context
    )

    prediction_df, context = (
        predict_bike_demand(
            weather_df,
            context=context,
        )
    )

    return prediction_df, context



# Historical Inference Smoke Test
# ============================================================
def run_smoke_test(test_date="2025-08-01"):
    """
    Validate the saved preprocessing/model artifacts by
    comparing inference predictions against predictions
    originally generated in modeling.py.
    """

    print("\n" + "=" * 60)
    print("Inference Smoke Test")
    print("=" * 60)

    prediction_timestamp = pd.Timestamp(
        test_date
    )

    prediction_date = (
        prediction_timestamp.date()
    )

    forecast_issued_at = datetime.combine(
        prediction_date - timedelta(days=1),
        time(20, 0),
        tzinfo=KST,
    )

    test_context = {
        "current_time": forecast_issued_at,
        "prediction_date": prediction_date,
        "forecast_issued_at": forecast_issued_at,
        "tmfc": forecast_issued_at.strftime("%Y%m%d%H"),
        "target_date": prediction_timestamp.strftime("%Y%m%d"),
    }

    print(
        f"Test Prediction Date: "
        f"{prediction_date}"
    )

    print(
        f"Forecast Issued At: "
        f"{forecast_issued_at}"
    )

    historical_df = pd.read_csv(
        DATASET_DIR
        / "seoul_bike_weather_forecast_data.csv",
        parse_dates=["date"],
    )

    historical_day_df = (
        historical_df[
            historical_df["date"]
            == prediction_timestamp
        ]
        .copy()
    )

    if (
        len(historical_day_df)
        != EXPECTED_DISTRICT_COUNT
    ):
        raise ValueError(
            f"Expected {EXPECTED_DISTRICT_COUNT} rows "
            f"for {test_date}, but found "
            f"{len(historical_day_df)}."
        )

    weather_df = (
        historical_day_df[
            WEATHER_COLUMNS
        ]
        .copy()
    )

    inference_predictions_df, _ = (
        predict_bike_demand(
            weather_df,
            context=test_context,
        )
    )

    print(
        "\nInference Prediction Shape: "
        f"{inference_predictions_df.shape}"
    )

    reference_path = (
        RESULTS_DIR
        / "modeling_test_predictions.csv"
    )

    if not reference_path.exists():
        raise FileNotFoundError(
            "Reference prediction file was not found. "
            "Run modeling.py first."
        )

    reference_df = pd.read_csv(
        reference_path,
        parse_dates=["date"],
    )

    reference_day_df = (
        reference_df[
            reference_df["date"]
            == prediction_timestamp
        ]
        .copy()
    )

    if (
        len(reference_day_df)
        != EXPECTED_DISTRICT_COUNT
    ):
        raise ValueError(
            f"Expected {EXPECTED_DISTRICT_COUNT} "
            f"reference rows for {test_date}, but found "
            f"{len(reference_day_df)}."
        )

    comparison_df = (
        inference_predictions_df[
            [
                "district",
                "predicted_use_count",
                "predicted_avg_use_time",
            ]
        ]
        .merge(
            reference_day_df[
                [
                    "district",
                    "use_count_pred",
                    "avg_use_time_pred",
                ]
            ],
            on="district",
            how="inner",
            validate="one_to_one",
        )
    )

    if (
        len(comparison_df)
        != EXPECTED_DISTRICT_COUNT
    ):
        raise ValueError(
            "District mismatch occurred during "
            "prediction comparison."
        )

    comparison_df[
        "use_count_difference"
    ] = (
        comparison_df[
            "predicted_use_count"
        ]
        - comparison_df[
            "use_count_pred"
        ]
    ).abs()

    comparison_df[
        "avg_use_time_difference"
    ] = (
        comparison_df[
            "predicted_avg_use_time"
        ]
        - comparison_df[
            "avg_use_time_pred"
        ]
    ).abs()

    use_count_match = np.allclose(
        comparison_df[
            "predicted_use_count"
        ],
        comparison_df[
            "use_count_pred"
        ],
        rtol=1e-10,
        atol=1e-8,
    )

    avg_use_time_match = np.allclose(
        comparison_df[
            "predicted_avg_use_time"
        ],
        comparison_df[
            "avg_use_time_pred"
        ],
        rtol=1e-10,
        atol=1e-8,
    )

    print("\n" + "-" * 60)

    print(
        "Maximum use_count difference:",
        comparison_df[
            "use_count_difference"
        ].max(),
    )

    print(
        "Maximum avg_use_time difference:",
        comparison_df[
            "avg_use_time_difference"
        ].max(),
    )

    print("\nPrediction Comparison")
    print(
        f"use_count match: "
        f"{use_count_match}"
    )
    print(
        f"avg_use_time match: "
        f"{avg_use_time_match}"
    )

    if (
        use_count_match
        and avg_use_time_match
    ):
        print(
            "\nSMOKE TEST PASSED:"
            "\nSaved models reproduce "
            "modeling.py predictions."
        )

    else:
        print(
            "\nSMOKE TEST FAILED:"
            "\nInference predictions differ "
            "from modeling.py predictions."
        )

    print("\nSample Comparison:")
    print(
        comparison_df
        .head()
        .round(6)
    )

    return comparison_df



# Manual Test
# ============================================================
if __name__ == "__main__":

    context = get_prediction_context()

    print("\n" + "=" * 60)
    print("Prediction Context")
    print("=" * 60)

    print(
        f"Current Time: "
        f"{context['current_time']}"
    )

    print(
        f"Forecast Issued At: "
        f"{context['forecast_issued_at']}"
    )

    print(
        f"Prediction Date: "
        f"{context['prediction_date']}"
    )

    print(
        f"KMA tmfc: "
        f"{context['tmfc']}"
    )

    run_smoke_test(
        test_date="2025-08-01"
    )
