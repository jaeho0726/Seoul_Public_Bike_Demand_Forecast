""" 
Past Weather Data Gathering
========================================================
1. 기상청 동네예보 격자 위경도 API 호출
2. 149 x 253 위경도 격자 생성
3. find_nearest_grid() 함수 정의
4. VWorld 시군구 SHP 파일 읽기
5. 서울 25개 구만 필터링
6. EPSG:5186 상태에서 centroid 계산
7. centroid를 위경도(EPSG:4326)로 변환
8. 각 자치구 centroid와 가장 가까운 KMA nx, ny 찾기
9. 최종 매핑 DataFrame 생성
10. CSV 저장
""" 

import requests
import numpy as np
import pandas as pd
import geopandas as gpd

import os
import time

from pathlib import Path
from datetime import datetime, timedelta


# =========================================================
# 0. 기본 설정
# =========================================================

Weather_API_Key = 'IPYr_3IPRKK2K_9yD3SiVg'

# VWorld에서 다운로드한 시군구 경계 SHP 파일 경로
shp_path = "./file/BND_SIGUNGU_PG.shp"

MAPPING_CSV = "seoul_district_kma_grid.csv"

CACHE_DIR = Path("kma_forecast_cache")
CACHE_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path("training_weather")
OUTPUT_DIR.mkdir(exist_ok=True)

GRID_X = 149
GRID_Y = 253
EXPECTED_SIZE = GRID_X * GRID_Y


# =========================================================
# 1. 기상청 격자 위경도 API 호출
# =========================================================

def get_kma_latlon_text(api_key):
    ## 기상청 단기예보(SHRT) 격자의 경도/위도 전체 배열을 text 형태로 받아온다.
    base_url = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_latlon_api"

    # 주의:
    # KMA API에서는 latlon=lon 응답이 실제 경도값,
    # latlon=lat 응답이 실제 위도값인지 반드시 실제 값 범위로 확인한다.

    lon_params = {
        "fct": "SHRT",
        "latlon": "lon",
        "disp": "A",
        "authKey": api_key
    }

    lat_params = {
        "fct": "SHRT",
        "latlon": "lat",
        "disp": "A",
        "authKey": api_key
    }

    lon_response = requests.get(
        base_url,
        params=lon_params,
        timeout=30
    )

    lat_response = requests.get(
        base_url,
        params=lat_params,
        timeout=30
    )

    lon_response.raise_for_status()
    lat_response.raise_for_status()

    return lon_response.text, lat_response.text


# =========================================================
# 2. KMA text 데이터를 숫자 배열로 변환
# =========================================================

def parse_kma_grid_text(text):
    ## KMA text/plain 응답에서 숫자만 추출한다.

    values = []

    for line in text.splitlines():
        for value in line.split(","):
            value = value.strip()

            if not value:
                continue

            try:
                values.append(float(value))
            except ValueError:
                pass

    return np.array(values)


# =========================================================
# 3. KMA 위경도 grid 생성
# =========================================================

lon_text, lat_text = get_kma_latlon_text(Weather_API_Key)

longitude_values = parse_kma_grid_text(lon_text)
latitude_values = parse_kma_grid_text(lat_text)

print("Raw longitude count:", len(longitude_values))
print("Raw latitude count:", len(latitude_values))


# KMA 응답 맨 앞에는 149, 253 격자 크기 정보가 포함되어 있음
expected_size = 149 * 253

if len(longitude_values) == expected_size + 2:
    print(
        "Longitude metadata 제거:",
        longitude_values[:2]
    )
    longitude_values = longitude_values[2:]

if len(latitude_values) == expected_size + 2:
    print(
        "Latitude metadata 제거:",
        latitude_values[:2]
    )
    latitude_values = latitude_values[2:]


# 최종 개수 검증
if len(longitude_values) != expected_size:
    raise ValueError(
        f"Longitude 예상 {expected_size}개, "
        f"실제 {len(longitude_values)}개"
    )

if len(latitude_values) != expected_size:
    raise ValueError(
        f"Latitude 예상 {expected_size}개, "
        f"실제 {len(latitude_values)}개"
    )


# 253 rows × 149 columns
longitude_grid = longitude_values.reshape(253, 149)

latitude_grid = latitude_values.reshape(253, 149)

print("Longitude grid shape:", longitude_grid.shape)
print("Latitude grid shape:", latitude_grid.shape)


# =========================================================
# 4. Haversine distance 함수
# =========================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    ## 두 위경도 사이의 거리를 km 단위로 계산한다.
    ## lat2, lon2는 NumPy array여도 됨.

    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return R * c


# =========================================================
# 5. 특정 위경도와 가장 가까운 KMA 격자 찾기
# =========================================================

def find_nearest_grid(
    target_lat,
    target_lon,
    latitude_grid,
    longitude_grid
):
    ## target_lat, target_lon과 가장 가까운 KMA nx, ny 격자를 반환한다.

    distance = haversine_distance(
        target_lat,
        target_lon,
        latitude_grid,
        longitude_grid
    )

    y_idx, x_idx = np.unravel_index(
        np.argmin(distance),
        distance.shape
    )

    return {
        "nx": int(x_idx + 1),
        "ny": int(y_idx + 1),
        "grid_lat": float(
            latitude_grid[y_idx, x_idx]
        ),
        "grid_lon": float(
            longitude_grid[y_idx, x_idx]
        ),
        "distance_km": float(
            distance[y_idx, x_idx]
        )
    }


# =========================================================
# 6. VWorld SHP 파일 불러오기
# =========================================================

gdf = gpd.read_file(shp_path)

print("\nOriginal CRS:")
print(gdf.crs)

print("\nColumns:")
print(gdf.columns)

print("\nTotal SIGUNGU count:")
print(len(gdf))


# =========================================================
# 7. 서울특별시 25개 구만 필터링
# =========================================================

# 현재 VWorld SIGUNGU_CD 구조에서 서울은 11로 시작
seoul_gdf = gdf[
    gdf["SIGUNGU_CD"]
    .astype(str)
    .str.startswith("11")
].copy()

print("\nSeoul district count:")
print(len(seoul_gdf))

print(
    seoul_gdf[
        ["SIGUNGU_CD", "SIGUNGU_NM"]
    ]
)


# 반드시 25개인지 검증
if len(seoul_gdf) != 25:
    raise ValueError(
        f"서울 자치구가 25개가 아닙니다. "
        f"현재 개수: {len(seoul_gdf)}"
    )


# =========================================================
# 8. 서울 자치구 geometric centroid 계산
# =========================================================

# SHP 원본 CRS = EPSG:5186
# 이미 meter 기반 projected CRS이므로 그대로 centroid 계산 가능

print("\nSeoul CRS:")
print(seoul_gdf.crs)

if seoul_gdf.crs is None:
    raise ValueError(
        "SHP 파일에 CRS 정보가 없습니다."
    )


seoul_gdf["centroid"] = (
    seoul_gdf.geometry.centroid
)


# centroid를 geometry로 사용하는 새로운 GeoDataFrame 생성
centroids = gpd.GeoDataFrame(
    seoul_gdf[
        [
            "BASE_DATE",
            "SIGUNGU_CD",
            "SIGUNGU_NM"
        ]
    ].copy(),
    geometry=seoul_gdf["centroid"],
    crs=seoul_gdf.crs
)


# =========================================================
# 9. centroid를 위도/경도 좌표계로 변환
# =========================================================

centroids = centroids.to_crs(
    epsg=4326
)

centroids["centroid_lon"] = (centroids.geometry.x)

centroids["centroid_lat"] = (centroids.geometry.y)

print("\nCentroid sample:")

print(
    centroids[
        [
            "SIGUNGU_NM",
            "centroid_lat",
            "centroid_lon"
        ]
    ].head()
)


# =========================================================
# 10. 각 서울 자치구 centroid -> KMA nx, ny 매핑
# =========================================================

mapping = []

for _, row in centroids.iterrows():

    result = find_nearest_grid(
        target_lat=row["centroid_lat"],
        target_lon=row["centroid_lon"],
        latitude_grid=latitude_grid,
        longitude_grid=longitude_grid
    )

    mapping.append({
        "district": row["SIGUNGU_NM"],
        "sigungu_cd": row["SIGUNGU_CD"],
        "base_date": row["BASE_DATE"],

        "centroid_lat": row["centroid_lat"],
        "centroid_lon": row["centroid_lon"],

        "nx": result["nx"],
        "ny": result["ny"],

        "grid_lat": result["grid_lat"],
        "grid_lon": result["grid_lon"],

        "distance_km": result["distance_km"]
    })


district_grid_df = pd.DataFrame(mapping)


# =========================================================
# 11. 결과 확인
# =========================================================

district_grid_df = (
    district_grid_df
    .sort_values("district")
    .reset_index(drop=True)
)

print("\nFinal district-grid mapping:")

print(
    district_grid_df[
        [
            "district",
            "centroid_lat",
            "centroid_lon",
            "nx",
            "ny",
            "grid_lat",
            "grid_lon",
            "distance_km"
        ]
    ]
)


# =========================================================
# 12. 매핑 거리 검증
# =========================================================

print("\n가장 먼 자치구 순:")

print(
    district_grid_df[
        [
            "district",
            "nx",
            "ny",
            "distance_km"
        ]
    ]
    .sort_values(
        "distance_km",
        ascending=False
    )
)


# =========================================================
# 13. 같은 KMA 격자를 공유하는 구 확인
# =========================================================

duplicate_grids = district_grid_df[
    district_grid_df.duplicated(
        subset=["nx", "ny"],
        keep=False
    )
].sort_values(
    ["nx", "ny"]
)

print("\n같은 KMA 격자를 공유하는 자치구:")

if len(duplicate_grids) == 0:
    print("없음")
else:
    print(
        duplicate_grids[
            [
                "district",
                "nx",
                "ny",
                "distance_km"
            ]
        ]
    )


# =========================================================
# 14. CSV 저장
# =========================================================

output_path = "seoul_district_kma_grid.csv"

district_grid_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"\n저장 완료: {output_path}"
)


# =========================================================
# 15. KMA 격자 응답 파싱
# =========================================================

def parse_forecast_grid(text):
    values = []

    for line in text.splitlines():
        for value in line.split(","):
            value = value.strip()

            if not value:
                continue

            try:
                values.append(float(value))
            except ValueError:
                continue

    values = np.array(values)

    if len(values) != EXPECTED_SIZE:
        raise ValueError(
            f"예상 격자값: {EXPECTED_SIZE}, 실제: {len(values)}"
        )

    return values.reshape(GRID_Y, GRID_X)


# =========================================================
# 16. KMA 결측값 처리
# =========================================================

def clean_kma_value(value):
    if value is None:
        return np.nan

    value = float(value)

    # KMA missing sentinel
    if value <= -90:
        return np.nan

    return value


# =========================================================
# 17. 캐시 파일 경로
# =========================================================

def get_cache_path(tmfc, tmef, variable):
    year = tmfc[:4]
    date = tmfc[:8]

    folder = CACHE_DIR / year / date
    folder.mkdir(parents=True, exist_ok=True)

    return folder / f"{variable}_{tmfc}_{tmef}.npy"


# =========================================================
# 18. 전국 예보 격자 1개 가져오기
# =========================================================

def get_forecast_grid(
    variable,
    tmfc,
    tmef,
    api_key,
    sleep_seconds=0.1
):
    ## 같은 요청은 cache에서 읽고, cache가 없을 때만 API 호출.

    cache_path = get_cache_path(
        tmfc,
        tmef,
        variable
    )

    ## 이미 받았으면 API 호출하지 않음
    if cache_path.exists():
        return np.load(cache_path)

    url = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_shrt_grd"

    params = {
        "tmfc": tmfc,
        "tmef": tmef,
        "vars": variable,
        "authKey": api_key
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    grid = parse_forecast_grid(
        response.text
    )

    ## 캐시 저장
    np.save(cache_path, grid)

    # API 과호출 방지
    time.sleep(sleep_seconds)

    return grid


# =========================================================
# 19. 서울 25개 구 mapping 읽기
# =========================================================

def load_district_mapping(mapping_csv):
    mapping_df = pd.read_csv(
        mapping_csv,
        dtype={
            "district": str,
            "sigungu_cd": str
        }
    )

    required_columns = {
        "district",
        "nx",
        "ny"
    }

    missing = (
        required_columns
        - set(mapping_df.columns)
    )

    if missing:
        raise ValueError(
            f"mapping CSV에 필요한 컬럼 없음: {missing}"
        )

    return mapping_df


# =========================================================
# 20. 전국 grid에서 서울 25개 구 값 동시 추출
# =========================================================

def extract_all_district_values(
    grid,
    mapping_df
):
    values = {}

    for _, row in mapping_df.iterrows():

        district = row["district"]

        nx = int(row["nx"])
        ny = int(row["ny"])

        value = grid[
            ny - 1,
            nx - 1
        ]

        values[district] = (
            clean_kma_value(value)
        )

    return values


# =========================================================
# 21. 하루치 시간별 변수 가져오기
# =========================================================

def collect_hourly_variable(
    variable,
    tmfc,
    target_date,
    mapping_df,
    api_key
):
    ## target_date의 00~23시 값을 받아 district별 list로 반환.

    district_values = {
        district: []
        for district in mapping_df["district"]
    }

    for hour in range(24):

        tmef = (
            f"{target_date}"
            f"{hour:02d}"
        )

        try:
            grid = get_forecast_grid(
                variable=variable,
                tmfc=tmfc,
                tmef=tmef,
                api_key=api_key
            )

        except Exception as e:
            print(
                f"[실패] {variable} "
                f"tmfc={tmfc}, "
                f"tmef={tmef}: {e}"
            )
            continue

        extracted = (
            extract_all_district_values(
                grid,
                mapping_df
            )
        )

        for district, value in extracted.items():

            if not np.isnan(value):
                district_values[
                    district
                ].append(value)

    return district_values


# =========================================================
# 22. TMX / TMN 유효값 찾기
# =========================================================

def collect_daily_temperature(
    variable,
    tmfc,
    target_date,
    mapping_df,
    api_key
):
    ## TMX / TMN은 모든 tmef에서 유효하지 않으므로 24시간을 돌면서 각 district의 첫 유효값을 저장.

    results = {
        district: np.nan
        for district in mapping_df["district"]
    }

    remaining = set(
        mapping_df["district"]
    )

    for hour in range(24):
        if not remaining:
            break

        tmef = (
            f"{target_date}"
            f"{hour:02d}"
        )

        try:
            grid = get_forecast_grid(
                variable=variable,
                tmfc=tmfc,
                tmef=tmef,
                api_key=api_key
            )

        except Exception as e:
            print(
                f"[실패] {variable} "
                f"tmef={tmef}: {e}"
            )
            continue

        extracted = (
            extract_all_district_values(
                grid,
                mapping_df
            )
        )

        for district in list(remaining):

            value = extracted[district]

            if not np.isnan(value):
                results[district] = value
                remaining.remove(district)

    return results


# =========================================================
# 23. 특정 날짜 하루치 서울 25구 feature 생성
# =========================================================

def get_all_districts_daily_forecast(
    target_date,
    api_key,
    mapping_csv=MAPPING_CSV
):
    """
    target_date = 실제 수요 예측 대상 날짜.

    예:
    target_date = 20240102

    사용 tmfc:
    20240101 20시 발표본
    """

    mapping_df = (
        load_district_mapping(
            mapping_csv
        )
    )

    target_dt = datetime.strptime(
        target_date,
        "%Y%m%d"
    )

    previous_day = (
        target_dt
        - timedelta(days=1)
    )

    tmfc = (
        previous_day.strftime(
            "%Y%m%d"
        )
        + "20"
    )

    print(
        f"\nTarget={target_date}, "
        f"TMFC={tmfc}"
    )

    # -----------------------------------------
    # TMX
    # -----------------------------------------

    tmx = collect_daily_temperature(
        variable="TMX",
        tmfc=tmfc,
        target_date=target_date,
        mapping_df=mapping_df,
        api_key=api_key
    )

    # -----------------------------------------
    # TMN
    # -----------------------------------------

    tmn = collect_daily_temperature(
        variable="TMN",
        tmfc=tmfc,
        target_date=target_date,
        mapping_df=mapping_df,
        api_key=api_key
    )

    # -----------------------------------------
    # REH
    # -----------------------------------------

    reh = collect_hourly_variable(
        variable="REH",
        tmfc=tmfc,
        target_date=target_date,
        mapping_df=mapping_df,
        api_key=api_key
    )

    # -----------------------------------------
    # POP
    # -----------------------------------------

    pop = collect_hourly_variable(
        variable="POP",
        tmfc=tmfc,
        target_date=target_date,
        mapping_df=mapping_df,
        api_key=api_key
    )

    # -----------------------------------------
    # WSD
    # -----------------------------------------

    wsd = collect_hourly_variable(
        variable="WSD",
        tmfc=tmfc,
        target_date=target_date,
        mapping_df=mapping_df,
        api_key=api_key
    )

    rows = []

    for _, row in mapping_df.iterrows():

        district = row["district"]

        reh_values = reh[district]
        pop_values = pop[district]
        wsd_values = wsd[district]

        rows.append({
            "date": target_date,
            "district": district,

            "tmfc": tmfc,

            "TMX": tmx[district],
            "TMN": tmn[district],

            "REH_mean": (
                np.mean(reh_values)
                if reh_values
                else np.nan
            ),

            "POP_max": (
                np.max(pop_values)
                if pop_values
                else np.nan
            ),

            "WSD_mean": (
                np.mean(wsd_values)
                if wsd_values
                else np.nan
            ),

            # 품질 확인용
            "REH_count": len(
                reh_values
            ),

            "POP_count": len(
                pop_values
            ),

            "WSD_count": len(
                wsd_values
            )
        })

    return pd.DataFrame(rows)


# =========================================================
# 24. 기간 전체 historical forecast 생성
# =========================================================

def get_historical_forecasts(
    start_date,
    end_date,
    api_key,
    mapping_csv=MAPPING_CSV,
    output_dir=OUTPUT_DIR
):
    """
    날짜별로 CSV를 저장하면서 진행.

    중간에 실행이 중단되어도
    이미 저장된 날짜는 skip.
    """

    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    all_data = []

    total = len(dates)

    for i, date in enumerate(
        dates,
        start=1
    ):

        target_date = (
            date.strftime(
                "%Y%m%d"
            )
        )

        daily_file = (
            output_dir
            / f"{target_date}.csv"
        )

        print(
            f"\n[{i}/{total}] "
            f"{target_date}"
        )

        # 이미 생성했으면 재사용
        if daily_file.exists():

            print(
                "이미 존재 → skip"
            )

            daily_df = pd.read_csv(
                daily_file
            )

            all_data.append(
                daily_df
            )

            continue

        try:

            daily_df = (
                get_all_districts_daily_forecast(
                    target_date=target_date,
                    api_key=api_key,
                    mapping_csv=mapping_csv
                )
            )

            # 날짜별 저장
            daily_df.to_csv(
                daily_file,
                index=False,
                encoding="utf-8-sig"
            )

            all_data.append(
                daily_df
            )

        except Exception as e:

            print(
                f"[날짜 전체 실패] "
                f"{target_date}: {e}"
            )

    if not all_data:
        return pd.DataFrame()

    final_df = pd.concat(
        all_data,
        ignore_index=True
    )

    return final_df


# =========================================================
# 25. 2024년 날씨 데이터 추출
# =========================================================
weather_2024 = get_historical_forecasts(
    start_date="2024-01-01",
    end_date="2024-12-31",
    api_key=Weather_API_Key
)

weather_2024.to_csv(
    "./dataset/weather_forecast_2024.csv",
    index=False,
    encoding="utf-8-sig"
)


