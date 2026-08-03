# Seoul Public Bike Demand Forecast Project

## File Structure

```
Seoul Public Bike Demand Forecast
|_ seoul_bike.py
|_ README.md
|_ dataset
    |_ 'seoul 2022-01-01 to 2024-01-01'.csv
    |_ seoul_bike_daily_data.csv
    |_ daily_data
        |_ Jan2022 ~ Dec_2023 (files for each 'month_year' between 2022 Jan ~ 2023 Dec)
            |_ bike_data_{year}{month}{day}.csv (each csv file has 25 rows, the number of state in Seoul)
```

## Dataset Source

- Seoul Open Data Plaza
  - [Seoul Public Bike (따릉이) Station Data](https://data.seoul.go.kr/dataList/OA-13252/F/1/datasetView.do)
  - [Seoul Public Bike (따릉이) Daily Usage Data](https://data.seoul.go.kr/dataList/OA-13252/F/1/datasetView.do)
- Kaggle
  - [Seoul Historical Weather Data](https://www.kaggle.com/datasets/alfredkondoro/seoul-historical-weather-data-2024?select=seoul+2022-01-01+to+2024-01-01.csv)
