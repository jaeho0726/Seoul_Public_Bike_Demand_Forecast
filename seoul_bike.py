import requests
import pandas as pd

API_Key = '444d424e786a61653930524d624872'

def get_daily_data(date):
  count_url = f'http://openapi.seoul.go.kr:8088/{API_Key}/json/tbCycleRentUseDayInfo/1/2/{date}'

  count_response = requests.get(count_url, timeout=30) 
  if count_response.status_code == 200:
    count_data = count_response.json()
  else: 
    return "Request Failed"
  
  total_count = int(count_data['cycleRentUseDayInfo']['list_total_count'])

  entire_rows = []
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
    start = end + 1
  
  data_df = pd.DataFrame(entire_rows)
  
  return data_df

print(get_daily_data('20230101'))
