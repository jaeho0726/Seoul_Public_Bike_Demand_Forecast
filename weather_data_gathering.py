import os
import time
import requests

import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime, timedelta


# =========================================================
# 0. 설정
# =========================================================

Weather_API_Key = os.getenv("KMA_API_KEY")

print("API key exists:", Weather_API_Key is not None)

if not Weather_API_Key:
    raise RuntimeError("KMA_API_KEY 환경변수가 설정되어 있지 않습니다.")

MAPPING_CSV = "./dataset/seoul_district_kma_grid.csv"

CACHE_DIR = Path("kma_forecast_cache")
CACHE_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path("./dataset/daily_weather")
OUTPUT_DIR.mkdir(exist_ok=True)

GRID_X = 149
GRID_Y = 253
EXPECTED_SIZE = GRID_X * GRID_Y


# =========================================================
# 1. KMA 격자 응답 파싱
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
# 2. KMA 결측값 처리
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
# 3. 전국 예보 격자 1개 가져오기
# =========================================================

def get_forecast_grid(
    variable,
    tmfc,
    tmef,
    api_key,
    sleep_seconds=0.2,
    max_retries=5
):
    cache_path = get_cache_path(
        tmfc,
        tmef,
        variable
    )

    if cache_path.exists():
        return np.load(cache_path)

    url = (
        "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/"
        "nph-dfs_shrt_grd"
    )

    params = {
        "tmfc": tmfc,
        "tmef": tmef,
        "vars": variable,
        "authKey": api_key
    }

    for attempt in range(1, max_retries + 1):

        try:
            response = requests.get(
                url,
                params=params,
                timeout=60
            )

            response.raise_for_status()

            grid = parse_forecast_grid(
                response.text
            )

            np.save(
                cache_path,
                grid
            )

            time.sleep(sleep_seconds)

            return grid

        except (
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError
        ) as e:

            print(
                f"[재시도 {attempt}/{max_retries}] "
                f"{variable} "
                f"tmfc={tmfc} "
                f"tmef={tmef}: {e}"
            )

            if attempt == max_retries:
                raise

            # 2, 4, 8, 16초...
            wait_time = 2 ** attempt

            print(
                f"{wait_time}초 후 재시도..."
            )

            time.sleep(wait_time)

    return grid


# =========================================================
# 4. 서울 25개 구 mapping 읽기
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
# 5. 전국 grid에서 서울 25개 구 값 동시 추출
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
# 6. 하루치 시간별 변수 가져오기
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
# 7. TMX / TMN 유효값 찾기
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
# 8. 특정 날짜 하루치 서울 25구 feature 생성
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
# 9. 기간 전체 historical forecast 생성
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

    time.sleep(2)


# =========================================================
# 10. 2024년 날씨 데이터 추출
# =========================================================
# weather_2024 = get_historical_forecasts(
#     start_date="2024-01-01",
#     end_date="2024-12-31",
#     api_key=Weather_API_Key
# )

# weather_2024.to_csv(
#     "./dataset/daily_weather_data/weather_forecast_2024.csv",
#     index=False,
#     encoding="utf-8-sig"
# )

# =========================================================
# 11. 2025년 날씨 데이터 추출
# =========================================================
weather_2025 = get_historical_forecasts(
    start_date="2025-01-01",
    end_date="2025-12-31",
    api_key=Weather_API_Key
)

weather_2025.to_csv(
    "./dataset/daily_weather_data/weather_forecast_2025.csv",
    index=False,
    encoding="utf-8-sig"
)