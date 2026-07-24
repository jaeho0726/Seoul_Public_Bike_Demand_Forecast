import requests

# API Key and URL for Seoul Bike Rental Data
API_Key = '444d424e786a61653930524d624872'
url = 'http://openapi.seoul.go.kr:8088/' + API_Key + '/json/tbCycleRentUseTimeInfo/1/5/20221002/9'

# Make a GET request to the API
response = requests.get(url)
if response.status_code == 200:
    data = response.json()

# Transform JSON data into a DataFrame
import pandas as pd
data_df = pd.DataFrame(data['cycleRentUseTimeInfo']['row'])

print(data_df.head())

