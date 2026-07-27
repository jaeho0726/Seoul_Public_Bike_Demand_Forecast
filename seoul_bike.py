import requests
import pandas as pd

API_Key = '444d424e786a61653930524d624872'

# Functions
## Functions of Loading Seoul City Bike Daily Usage Data 
def get_total_count(date):
    count_url = f'http://openapi.seoul.go.kr:8088/{API_Key}/json/tbCycleRentUseDayInfo/1/2/{date}'

    count_response = requests.get(count_url, timeout=30)
    if count_response.status_code == 200:
        count_data = count_response.json()
    else:
        return "Request Failed"

    total_count = int(count_data['CycleRentUseDayInfo']['list_total_count'])
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

    # Dropping unnecessary columns
    data_df_cleaned = data_df.drop(columns = ['EXER_AMT', 'CARBON_AMT', 'RENT_TYPE', 'GENDER_CD', 'START_INDEX', 'END_INDEX', 'RNUM'])

    # Changing the integer format of 'RENT_ID' 
    data_df_cleaned['rentid'] = data_df_cleaned['RENT_NM'].str.split(".")
    data_df_cleaned['rentid'] = data_df_cleaned['rentid'].str[0]
    data_df_cleaned['RENT_ID'] = data_df_cleaned['rentid']
    data_df_cleaned = data_df_cleaned.drop(columns = ['RENT_NM', 'rentid'])
    
    return data_df_cleaned

## Functions of Loading Seoul City Bike Station Data 
def get_station_count():
  url = f'http://openapi.seoul.go.kr:8088/{API_Key}/json/tbCycleStationInfo/1/2/'
  count_response = requests.get(url, timeout=30)

  if count_response.status_code == 200:
          count_station_data = count_response.json()
  else:
      return "Request Failed"

  total_count = int(count_station_data['stationInfo']['list_total_count'])
  return total_count

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