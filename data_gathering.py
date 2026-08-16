# =========================================================
# Seoul Public Bike Demand Forecast - Data Gathering
# Description: This script gathers daily bike usage data from the Seoul Open API, processes it, and saves it into CSV files for further analysis. 
# =========================================================

# Importing necessary libraries
# =========================================================
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
from pathlib import Path
from holidayskr import is_holiday
import os


API_Key = os.getenv('Seoul_Open_Data_API_Key')

if not API_Key:
    raise RuntimeError(
        "Missing SEOUL_OPEN_DATA_API_KEY environment variable. "
        "Set it before running this script."
    )

# Functions
# =========================================================
## Functions of Loading Seoul City Bike Daily Usage Data 
def get_total_count(date):
    count_url = f'http://openapi.seoul.go.kr:8088/{API_Key}/json/tbCycleRentUseDayInfo/1/2/{date}'

    count_response = requests.get(count_url, timeout=30)
    if count_response.status_code == 200:
        count_data = count_response.json()
    else:
        return "Request Failed"

    total_count = int(count_data['cycleRentUseDayInfo']['list_total_count'])
    return total_count

def get_daily_data(date):
    entire_rows = []

    total_count = get_total_count(date)
    start = 1
    page_size = 1000

    while start <= total_count:
        end = start + page_size - 1
        if end > total_count:
            end = total_count

        url = f'http://openapi.seoul.go.kr:8088/{API_Key}/json/tbCycleRentUseDayInfo/{start}/{end}/{date}'

        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
        else:
            return "Request Failed"

        rows = data['cycleRentUseDayInfo']['row']
        entire_rows.extend(rows)
        start += page_size

    data_df = pd.DataFrame(entire_rows)

    ### Dropping unnecessary columns
    data_df_cleaned = data_df.drop(columns = ['EXER_AMT', 'CARBON_AMT', 'RENT_TYPE', 'GENDER_CD', 'START_INDEX', 'END_INDEX', 'RNUM', 'MOVE_METER'])

    ### Changing the integer format of 'RENT_ID' column
    data_df_cleaned['rentid'] = data_df_cleaned['RENT_NM'].str.split(".")
    data_df_cleaned['rentid'] = data_df_cleaned['rentid'].str[0]
    data_df_cleaned['RENT_ID'] = data_df_cleaned['rentid']
    data_df_cleaned = data_df_cleaned.drop(columns = ['RENT_NM', 'rentid'])

    ### Changing data type of 'USE_CNT' and 'MOVE_TIME' columns
    data_df_cleaned = data_df_cleaned.astype({'USE_CNT' : 'int64', 'MOVE_TIME' : 'int64'})

    ### Merging with 'station_data' based on Station ID
    station_data = get_station_data()
    data_df_merged = pd.merge(data_df_cleaned, station_data, on = 'RENT_ID', how = 'left')

    ### Grouping based on the state
    data_df_merged = data_df_merged.groupby(['STA_LOC'])[['USE_CNT', 'MOVE_TIME']].agg({'USE_CNT': 'sum', 'MOVE_TIME': 'sum'})

    ### Calculating average usage time per rental
    data_df_merged['AVG_MOVE_TIME'] = (
        data_df_merged['MOVE_TIME']
        / data_df_merged['USE_CNT']
    )
    data_df_merged = data_df_merged.astype({'AVG_MOVE_TIME' : 'float64'})
    data_df_merged = data_df_merged.drop(columns = ['MOVE_TIME'])

    ### Adding date, day of week, holiday columns
    date_timestamp = pd.to_datetime(date, format="%Y%m%d")
    date_string = date_timestamp.strftime("%Y-%m-%d")

    data_df_merged['DATE'] = date_timestamp
    data_df_merged['Day_of_Week'] = date_timestamp.day_name()
    data_df_merged['Is_Holiday'] = bool(is_holiday(date_string))

    data_df_merged["Is_Weekend"] = (data_df_merged['DATE'].dt.dayofweek >= 5)

    ### Reseting the index to later merge with weather data
    data_df_merged = data_df_merged.reset_index()
    
    weather_data = get_weather_data()
    data_df_weather_merged = data_df_merged.merge(weather_data, on='DATE', how='left')

    return data_df_weather_merged

## Function of Loading Seoul City Bike Station Data 
def get_station_count():
  url = f'http://openapi.seoul.go.kr:8088/{API_Key}/json/tbCycleStationInfo/1/2/'
  count_response = requests.get(url, timeout=30)

  if count_response.status_code == 200:
          count_station_data = count_response.json()
  else:
      return "Request Failed"

  total_count = int(count_station_data['stationInfo']['list_total_count'])
  return total_count

## Function of Loading Seoul City Bike Station Data
def get_station_data():
  entire_station = []

  num_station = get_station_count()
  start = 1
  page_size = 1000

  while start <= num_station:
      end = start + page_size - 1
      if end > num_station:
          end = num_station

      url = f'http://openapi.seoul.go.kr:8088/{API_Key}/json/tbCycleStationInfo/{start}/{end}/'

      response = requests.get(url, timeout=30)
      if response.status_code == 200:
          data = response.json()
      else:
          return "Request Failed"

      rows = data['stationInfo']['row']
      entire_station.extend(rows)
      start += page_size

  station_data_df = pd.DataFrame(entire_station)

  station_data_df_cleaned = station_data_df[['STA_LOC', 'RENT_ID_NM']]
  station_data_df_cleaned['RENT_ID'] = station_data_df_cleaned['RENT_ID_NM'].str.split(".")
  station_data_df_cleaned['RENT_ID'] = station_data_df_cleaned['RENT_ID'].str[0]
  station_data_df_cleaned = station_data_df_cleaned.drop(columns = ['RENT_ID_NM'])

  return station_data_df_cleaned

## Function of Loading Seoul Weather Data
def get_weather_data():
    weather_data = pd.read_csv('./dataset/seoul 2022-01-01 to 2024-01-01.csv')
    weather_data_cleaned = weather_data[['datetime', 'tempmax', 'tempmin', 'feelslike', 'humidity', 'precip']]
    weather_data_cleaned['DATE'] = pd.to_datetime(weather_data_cleaned['datetime'], format='%Y-%m-%d')
    weather_data_cleaned = weather_data_cleaned.drop(columns = ['datetime'])
    weather_data_cleaned = weather_data_cleaned.astype({'tempmax' : 'float64', 'tempmin' : 'float64', 'feelslike' : 'float64', 'humidity' : 'float64', 'precip' : 'float64'})
    return weather_data_cleaned

## Function of Creating an Ultimate DataFrame Containing All Daily Data of 2024
save_dir = Path("dataset/daily_data")
save_dir.mkdir(parents=True, exist_ok=True)

dates_2024 = pd.date_range(
    start="2022-08-31",
    end="2023-11-23",
    freq="D"
)

failed_dates = []

for date in dates_2024: 
    date_str = date.strftime("%Y%m%d")

    ### Creating folder name 
    month_folder = date.strftime("%b_%Y")
    (save_dir / month_folder).mkdir(parents=True, exist_ok=True)

    file_path = save_dir / month_folder / f"bike_data_{date_str}.csv"
    

    ### Skip dates that were already downloaded
    if file_path.exists():
        print(f"{date_str}: already saved")
        continue

    try:
        daily_df = get_daily_data(date_str)

        ### get_daily_data() currently returns a string when a request fails
        if not isinstance(daily_df, pd.DataFrame):
            print(f"{date_str}: failed - {daily_df}")
            failed_dates.append(date_str)
            continue

        if daily_df.empty:
            print(f"{date_str}: empty DataFrame")
            failed_dates.append(date_str)
            continue

        daily_df.to_csv(
            file_path,
            index=False
        )

        print(
            f"{date_str}: saved "
            f"{len(daily_df):,} rows"
        )

        ### Remove the daily DataFrame from memory
        del daily_df

        ### Avoid sending requests too quickly
        time.sleep(0.1)

    except Exception as error:
        print(f"{date_str}: error - {error}")
        failed_dates.append(date_str)