# ============================================================
# Seoul Public Bike Demand Forecast - Dashboard
# ============================================================
import json
import os

from pathlib import Path

import plotly.express as px
import streamlit as st

from inference import (
    get_prediction_context,
    run_live_inference,
)



# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Seoul Public Bike Demand Forecast",
    page_icon="🚲",
    layout="wide",
)



# Custom CSS
# ============================================================
st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Forecast for / Last updated labels */
    div[data-testid="stMetricLabel"] p {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }

    /* Forecast date/time values */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 650 !important;
    }

    /* View label */
    .st-key-metric_toggle label p {
        font-size: 1.35rem !important;
        font-weight: 650 !important;
    }

    /* Segmented control */
    .st-key-metric_toggle button {
        font-size: 1.05rem !important;
        min-height: 3.2rem !important;
        border-color: #4A5A50 !important;
    }

    /* Selected option */
    .st-key-metric_toggle
    button[aria-pressed="true"] {
        background: #2E8B57 !important;
        background-color: #2E8B57 !important;
        border-color: #2E8B57 !important;
        color: white !important;
    }

    .st-key-metric_toggle
    button[aria-pressed="true"] p {
        color: white !important;
    }

    /* Unselected option */
    .st-key-metric_toggle
    button[aria-pressed="false"] {
        background: transparent !important;
        color: #DDE8E1 !important;
        border-color: #4A5A50 !important;
    }

    .st-key-metric_toggle
    button[aria-pressed="false"] p {
        color: #DDE8E1 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)



# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

GEOJSON_PATH = (BASE_DIR/"dataset"/"seoul_districts.geojson")



# Streamlit Cloud Secret Support
# ============================================================
def configure_kma_api_key():
    """
    Use KMA_API_KEY from the environment when available.

    For Streamlit Community Cloud, also support a secret
    named KMA_API_KEY in .streamlit/secrets.toml or the
    deployment Secrets settings.
    """

    if os.getenv("KMA_API_KEY"):
        return

    try:
        secret_key = st.secrets.get(
            "KMA_API_KEY",
            None,
        )
    except Exception:
        secret_key = None

    if secret_key:
        os.environ["KMA_API_KEY"] = str(
            secret_key
        )


configure_kma_api_key()



# Load GeoJSON
# ============================================================
@st.cache_data
def load_geojson():

    with open(
        GEOJSON_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


seoul_geojson = load_geojson()



# Live Prediction
# ============================================================
context = get_prediction_context()


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def load_live_predictions(
    tmfc,
    target_date,
):
    """
    Retrieve KMA forecast data and generate the 25-district
    model predictions.

    tmfc and target_date are cache keys, so after the 20:00
    KST forecast switch Streamlit automatically uses a new
    cache entry.
    """

    prediction_context = (
        get_prediction_context()
    )

    # Guarantee that the inference call uses the exact
    # forecast identifiers represented by this cache entry.
    prediction_context["tmfc"] = tmfc
    prediction_context[
        "target_date"
    ] = target_date

    predictions, prediction_context = (
        run_live_inference(
            context=prediction_context
        )
    )

    return (
        predictions,
        prediction_context,
    )


try:
    with st.spinner(
        "Loading the latest Seoul bike forecast..."
    ):
        predictions, context = (
            load_live_predictions(
                tmfc=context["tmfc"],
                target_date=context[
                    "target_date"
                ],
            )
        )

except Exception as exc:
    st.error(
        "The latest forecast could not be generated."
    )

    st.caption(
        "Check the KMA API key, network connection, "
        "and availability of the selected KMA forecast."
    )

    with st.expander(
        "Technical details"
    ):
        st.exception(exc)

    st.stop()



# Validate GeoJSON / Prediction District Match
# ============================================================
geojson_districts = {
    feature["properties"][
        "SIG_KOR_NM"
    ]
    for feature in seoul_geojson[
        "features"
    ]
}

prediction_districts = set(
    predictions["district"]
    .astype(str)
    .str.strip()
)

if (
    geojson_districts != prediction_districts
):
    missing_from_predictions = (
        geojson_districts
        - prediction_districts
    )

    missing_from_geojson = (
        prediction_districts
        - geojson_districts
    )

    st.error(
        "District names in the prediction output "
        "do not match the Seoul GeoJSON."
    )

    st.write(
        "Missing from predictions:",
        sorted(
            missing_from_predictions
        ),
    )

    st.write(
        "Missing from GeoJSON:",
        sorted(
            missing_from_geojson
        ),
    )

    st.stop()



# Display Metadata
# ============================================================
prediction_date = (
    context["prediction_date"]
)

forecast_issued_at = (
    context["forecast_issued_at"]
)

generated_at = (
    context["generated_at"]
)

now_kst = (
    context["current_time"]
)

display_prediction_date = (
    prediction_date.strftime(
        "%B %d, %Y"
    )
)

display_update_time = (
    generated_at.strftime(
        "%B %d, %Y · %H:%M KST"
    )
)


if now_kst.hour >= 20:
    forecast_note = (
        "Tomorrow's prediction uses the "
        "**20:00 KST KMA weather forecast**."
    )

else:
    forecast_note = (
        "Tomorrow's prediction becomes available "
        "after **20:00 KST**. "
        "Currently showing today's latest forecast."
    )



# Information Dialog
# ============================================================
@st.dialog(
    "About this forecast"
)
def show_forecast_info():

    st.markdown(
        f"""
        **Forecast timing**

        **Before 20:00 KST**  
        The dashboard displays the latest available
        prediction for **today**.

        **After 20:00 KST**  
        It switches to **tomorrow's prediction** using
        the KMA weather forecast issued at 20:00 KST.

        **Weather forecast issued**  
        {forecast_issued_at.strftime("%B %d, %Y · %H:%M KST")}

        ---

        **Rental Demand**  
        Predicted total daily Seoul Public Bike rentals
        in each district.

        **Average Use Time**  
        Predicted average rental duration in each district.
        """
    )



# Header
# ============================================================
title_col, info_col = st.columns(
    [15, 1],
    vertical_alignment="center",
)

with title_col:
    st.title(
        "Seoul Public Bike Demand Forecast"
    )

with info_col:
    if st.button(
        "ⓘ",
        key="forecast_info",
        help="About this forecast",
    ):
        show_forecast_info()



# Forecast Information
# ============================================================
date_col, updated_col = (
    st.columns(2)
)

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

st.caption(
    forecast_note
)

st.divider()



# Metric Selector
# ============================================================
selected_metric = (
    st.segmented_control(
        "View",
        options=[
            "Rental Demand",
            "Average Use Time",
        ],
        default="Rental Demand",
        required=True,
        key="metric_toggle",
    )
)


if (
    selected_metric
    == "Rental Demand"
):
    color_column = (
        "predicted_use_count"
    )

    ranking_column = (
        "predicted_use_count"
    )

    color_label = (
        "Predicted Rentals"
    )

    ranking_title = (
        "Highest Predicted Demand"
    )

else:
    color_column = (
        "predicted_avg_use_time"
    )

    ranking_column = (
        "predicted_avg_use_time"
    )

    color_label = (
        "Average Use Time (min)"
    )

    ranking_title = (
        "Longest Predicted Use Time"
    )



# Seoul Bike Green Scale
# ============================================================
SEOUL_BIKE_GREEN_SCALE = [
    "#E3F2E6",
    "#B8DEBF",
    "#7FC88A",
    "#45A85B",
    "#1C7A3A",
    "#095C2A",
]



# Build Choropleth
# ============================================================
fig = px.choropleth(
    predictions,
    geojson=seoul_geojson,
    locations="district",
    featureidkey=(
        "properties.SIG_KOR_NM"
    ),
    color=color_column,
    color_continuous_scale=(
        SEOUL_BIKE_GREEN_SCALE
    ),
    custom_data=[
        "district",
        "predicted_use_count",
        "predicted_avg_use_time",
    ],
    labels={
        "predicted_use_count":
            "Predicted Rentals",
        "predicted_avg_use_time":
            "Average Use Time",
    },
)


fig.update_geos(
    fitbounds="locations",
    visible=False,
)


fig.update_traces(
    hovertemplate=(
        "<b>%{customdata[0]}</b><br>"
        "Predicted rentals: "
        "%{customdata[1]:,.0f}<br>"
        "Avg. use time: "
        "%{customdata[2]:.1f} min"
        "<extra></extra>"
    ),
    marker_line_color="#111111",
    marker_line_width=1.2,
)


fig.update_layout(
    margin=dict(
        l=0,
        r=0,
        t=0,
        b=0,
    ),
    coloraxis_colorbar=dict(
        title=color_label,
        thickness=15,
    ),
    height=650,
    paper_bgcolor=(
        "rgba(0,0,0,0)"
    ),
    plot_bgcolor=(
        "rgba(0,0,0,0)"
    ),
)



# Main Layout
# ============================================================
map_col, ranking_col = (
    st.columns([3, 1])
)


with map_col:
    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displayModeBar": False,
            "scrollZoom": False,
        },
    )


# Top 3 Districts
# ============================================================
with ranking_col:

    st.subheader(
        ranking_title
    )

    top3 = (
        predictions
        .nlargest(
            3,
            ranking_column,
        )
        .reset_index(
            drop=True
        )
    )

    for index, row in (
        top3.iterrows()
    ):
        rank = index + 1

        if (
            selected_metric
            == "Rental Demand"
        ):
            value_text = (
                f"{row['predicted_use_count']:,.0f} "
                f"predicted rentals"
            )

        else:
            value_text = (
                f"{row['predicted_avg_use_time']:.1f} "
                f"min average"
            )

        st.markdown(
            f"""
            ### {rank}. {row['district']}
            **{value_text}**
            """
        )

        if index < 2:
            st.divider()
