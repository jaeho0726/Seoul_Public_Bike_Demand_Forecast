# =========================================================
# Seoul Public Bike Demand Forecast - Preprocessing
# =========================================================

# Importing necessary libraries
# =========================================================
import pandas as pd
from pathlib import Path



# Combining daily CSV files into one CSV file
# =========================================================

def combine_daily_csv_files():

    ## Parent folder containing Jan_2022, Feb_2022, ..., Dec_2023
    daily_data_dir = Path("dataset/daily_data")

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
        "Is_Weekend": "is_weekend",
        "tempmax": "temp_max",
        "tempmin": "temp_min",
        "feelslike": "feels_like",
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


combined_bike_df = combine_daily_csv_files()



# Data Checking
# =========================================================

## Checking Data Types
print("\nData Types:")
print(combined_bike_df.columns.tolist())
print(combined_bike_df.dtypes)

## Checking Null Values
print("\nNull Values:")
print(combined_bike_df.isnull().sum())
