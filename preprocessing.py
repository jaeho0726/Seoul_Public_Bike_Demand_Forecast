# =========================================================
# Seoul Public Bike Demand Forecast - Preprocessing
# =========================================================

# Importing necessary libraries
# =========================================================
import pandas as pd
from pathlib import Path



# Combining daily bike usage CSV files into one CSV file
# =========================================================

def combine_daily_bike_data():

    ## Parent folder containing Jan_2024, Feb_2024, ..., Dec_2024
    daily_data_dir = Path("dataset/daily_bike_data")

    ## Final combined CSV location
    output_file = Path("dataset/seoul_bike_daily_data.csv")

    ## Find CSV files inside all month folders
    csv_files = sorted(daily_data_dir.rglob("bike_data_*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No daily CSV files were found inside: {daily_data_dir.resolve()}"
        )

    dataframes = []

    for file_path in csv_files:
        try:
            daily_df = pd.read_csv(file_path, parse_dates=["DATE"])

            if daily_df.empty:
                print(f"Skipped empty file: {file_path}")
                continue

            ## Validating whether each file contains 25 Seoul districts
            if len(daily_df) != 25:
                print(
                    f"Warning: {file_path.name} contains "
                    f"{len(daily_df)} rows instead of 25."
                )

            dataframes.append(daily_df)

        except Exception as error:
            print(f"Could not read {file_path}: {error}")

    if not dataframes:
        raise ValueError("No valid CSV files were available to combine.")

    ## Combine all daily DataFrames vertically
    combined_df = pd.concat(dataframes, ignore_index=True)

    ## Sort chronologically and then by district
    sort_columns = ["DATE"]

    if "STA_LOC" in combined_df.columns:
        sort_columns.append("STA_LOC")

    combined_df = combined_df.sort_values(
        by=sort_columns
    ).reset_index(drop=True)

    ## Check for invalid DATE values
    invalid_dates = combined_df["DATE"].isna().sum()

    if invalid_dates > 0:
        print(f"Warning: {invalid_dates} rows have invalid DATE values.")

    columns_rename_map = {
        "STA_LOC": "district",
        "USE_CNT": "use_count",
        "AVG_MOVE_TIME": "avg_use_time",
        "DATE": "date",
        "Day_of_Week": "day_of_week",
        "Is_Holiday": "is_holiday",
        "Is_Weekend": "is_weekend"
        }
    
    combined_df = combined_df.rename(columns=columns_rename_map)

    ## Save as one CSV
    combined_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nCombination completed.")
    print(f"Number of CSV files combined: {len(dataframes):,}")
    print(f"Total rows: {len(combined_df):,}")
    print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"Saved to: {output_file.resolve()}")

    return combined_df


combined_bike_df = combine_daily_bike_data()



# Bike Data Checking
# =========================================================

## Checking Data Types
print("\nData Types:")
print(combined_bike_df.columns.tolist())
print(combined_bike_df.dtypes)

## Checking Null Values
print("\nNull Values:")
print(combined_bike_df.isnull().sum())



# Combining daily weather CSV files into one CSV file
# =========================================================

def combine_weather_forecast_data():
    weather_files = [
        Path("dataset/weather_forecast_2024.csv"),
        Path("dataset/weather_forecast_2025.csv")
    ]

    output_file = Path("dataset/seoul_weather_daily_data.csv")


    weather_dataframes = []

    for file_path in weather_files:

        # Check whether the file exists
        if not file_path.exists():
            raise FileNotFoundError(
                f"Weather file not found: {file_path.resolve()}"
            )

        try:
            weather_df = pd.read_csv(file_path)

            if weather_df.empty:
                raise ValueError(
                    f"Weather file is empty: {file_path}"
                )

            weather_dataframes.append(weather_df)

        except Exception as error:
            raise RuntimeError(
                f"Could not read {file_path}: {error}"
            )


    # Combine 2024 and 2025 weather data
    combined_weather_df = pd.concat(
        weather_dataframes,
        ignore_index=True
    )

    # Converting 'DATE' column to datetime
    combined_weather_df["date"] = pd.to_datetime(
        combined_weather_df["date"].astype(str),
        format="%Y%m%d"
    )

    # Sort by date and column
    combined_weather_df = (
        combined_weather_df
        .sort_values(
            by=[
                "date",
                "district",
            ]
        )
        .reset_index(drop=True)
    )

    ## Save as one CSV
    combined_weather_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    return combined_weather_df

combined_weather_df = combine_weather_forecast_data()



# Weather Data Checking
# =========================================================

print(
    f"Total rows: "
    f"{len(combined_weather_df):,}"
)

# Check duplicate date × district combinations
duplicate_count = (
    combined_weather_df
    .duplicated(
        subset=[
            "date",
            "district"
        ]
    )
    .sum()
)

print(
    f"Duplicate date-district rows: "
    f"{duplicate_count}"
)


# Check null values
print("\nNull Values:")

print(
    combined_weather_df
    .isnull()
    .sum()
)