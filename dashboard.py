from pathlib import Path
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Seoul Public Bike Demand Forecast",
    page_icon="🚲",
    layout="wide",
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
GEOJSON_PATH = BASE_DIR / "dataset" / "seoul_districts.geojson"


# ============================================================
# Load GeoJSON
# ============================================================

@st.cache_data
def load_geojson():
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


seoul_geojson = load_geojson()


# ============================================================
# Temporary dummy predictions
# Replace this later with inference.py output
# ============================================================

@st.cache_data
def create_dummy_predictions(geojson):
    districts = sorted(
        feature["properties"]["SIG_KOR_NM"]
        for feature in geojson["features"]
    )

    # Fixed random seed so values do not change every refresh
    rng = np.random.default_rng(42)

    predicted_use_count = rng.integers(
        low=2500,
        high=9500,
        size=len(districts),
    )

    predicted_avg_use_time = rng.uniform(
        low=15.0,
        high=30.0,
        size=len(districts),
    )

    return pd.DataFrame(
        {
            "district": districts,
            "predicted_use_count": predicted_use_count,
            "predicted_avg_use_time": predicted_avg_use_time,
        }
    )


predictions = create_dummy_predictions(seoul_geojson)


# ============================================================
# Forecast metadata
# Temporary logic for UI development
# ============================================================

KST = ZoneInfo("Asia/Seoul")
now_kst = datetime.now(KST)

# For dashboard prototyping only:
# after 20:00 → tomorrow
# before 20:00 → today's latest available forecast
if now_kst.hour >= 20:
    prediction_date = now_kst.date() + timedelta(days=1)
else:
    prediction_date = now_kst.date()

display_prediction_date = prediction_date.strftime("%B %d, %Y")
display_update_time = now_kst.strftime("%B %d, %Y · %H:%M KST")


# ============================================================
# Header
# ============================================================

title_col, info_col = st.columns([12, 1])

with title_col:
    st.title("Seoul Public Bike Demand Forecast")

with info_col:
    with st.popover("ⓘ"):
        st.markdown(
            """
            ### About this forecast

            This dashboard displays predicted Seoul Public Bike usage
            across Seoul's 25 districts.

            Predictions include:

            - **Daily rental demand**
            - **Average rental duration**

            Weather features are based on the Korea Meteorological
            Administration forecast issued at **20:00 KST on the
            previous day**.

            New next-day predictions become available after the
            required weather forecast has been retrieved and processed.
            """
        )


# ============================================================
# Forecast information
# ============================================================

date_col, updated_col = st.columns(2)

with date_col:
    st.metric(
        label="Forecast for",
        value=display_prediction_date,
    )

with updated_col:
    st.metric(
        label="Last updated",
        value=display_update_time,
    )


st.divider()


# ============================================================
# Metric selector
# ============================================================

selected_metric = st.radio(
    "View",
    options=["Rental Demand", "Average Use Time"],
    horizontal=True,
)


if selected_metric == "Rental Demand":
    color_column = "predicted_use_count"
    ranking_column = "predicted_use_count"
    color_label = "Predicted Rentals"
    ranking_title = "Highest Predicted Demand"

else:
    color_column = "predicted_avg_use_time"
    ranking_column = "predicted_avg_use_time"
    color_label = "Average Use Time (min)"
    ranking_title = "Longest Predicted Use Time"


# ============================================================
# Build map
# ============================================================

fig = px.choropleth(
    predictions,
    geojson=seoul_geojson,
    locations="district",
    featureidkey="properties.SIG_KOR_NM",
    color=color_column,
    color_continuous_scale="YlOrRd",
    custom_data=[
        "district",
        "predicted_use_count",
        "predicted_avg_use_time",
    ],
    labels={
        "predicted_use_count": "Predicted Rentals",
        "predicted_avg_use_time": "Average Use Time",
    },
)


# Automatically zoom to Seoul district boundaries
fig.update_geos(
    fitbounds="locations",
    visible=False,
)


# Custom hover content
fig.update_traces(
    hovertemplate=(
        "<b>%{customdata[0]}</b><br>"
        "Predicted rentals: %{customdata[1]:,.0f}<br>"
        "Avg. use time: %{customdata[2]:.1f} min"
        "<extra></extra>"
    ),
    marker_line_color="white",
    marker_line_width=1,
)


fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    coloraxis_colorbar=dict(
        title=color_label,
        thickness=15,
    ),
    height=650,
)


# ============================================================
# Main dashboard layout
# ============================================================

map_col, ranking_col = st.columns([3, 1])


with map_col:
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "scrollZoom": False,
        },
    )


# ============================================================
# Top 3 districts
# ============================================================

with ranking_col:
    st.subheader(ranking_title)

    top3 = predictions.nlargest(
        3,
        ranking_column,
    ).reset_index(drop=True)

    for index, row in top3.iterrows():

        rank = index + 1

        if selected_metric == "Rental Demand":
            value_text = (
                f"{row['predicted_use_count']:,.0f} predicted rentals"
            )

        else:
            value_text = (
                f"{row['predicted_avg_use_time']:.1f} min average"
            )

        st.markdown(
            f"""
            ### {rank}. {row['district']}
            **{value_text}**
            """
        )

        if index < 2:
            st.divider()