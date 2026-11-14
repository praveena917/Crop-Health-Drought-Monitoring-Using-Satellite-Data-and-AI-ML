
import streamlit as st
import pandas as pd
import plotly.express as px
import folium

from streamlit_folium import st_folium

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="India Crop Health Monitoring",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 India-Wide Crop Health & Environmental Anomaly Monitoring")

st.markdown(
    """
    Satellite-based vegetation monitoring combined with
    NASA POWER weather data and Isolation Forest anomaly detection.
    """
)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/processed/india_anomaly_results.csv"
    )

    return df


df = load_data()

# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.header("Dashboard Filters")

years = sorted(df["year"].unique())

selected_years = st.sidebar.multiselect(
    "Select years",
    years,
    default=years
)

locations = sorted(df["location_id"].unique())

selected_locations = st.sidebar.multiselect(
    "Select locations",
    locations,
    default=locations
)

admin_units = sorted(df["admin_unit"].unique())

selected_admin = st.sidebar.multiselect(
    "Select administrative units",
    admin_units,
    default=admin_units
)

filtered = df[
    df["year"].isin(selected_years)
    & df["location_id"].isin(selected_locations)
    & df["admin_unit"].isin(selected_admin)
].copy()

if filtered.empty:

    st.warning("No observations match the selected filters.")

    st.stop()

# ==========================================
# KPI METRICS
# ==========================================

st.subheader("Environmental Monitoring Overview")

total = len(filtered)

unusual = (
    filtered["environmental_status"] == "Unusual"
).sum()

normal = total - unusual

unusual_percentage = (
    unusual / total * 100 if total else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Observations", total)

col2.metric("Locations", filtered["location_id"].nunique())

col3.metric("Normal", normal)

col4.metric(
    "Unusual",
    unusual,
    f"{unusual_percentage:.1f}% of observations"
)

st.caption(
    "Unusual means flagged by an unsupervised anomaly-detection model. "
    "It does not confirm drought."
)

st.divider()

# ==========================================
# INDIA SAMPLE LOCATION MAP
# ==========================================

st.subheader("India Sample Location Map")

map_data = (
    filtered.groupby(
        ["location_id", "admin_unit", "latitude", "longitude"],
        as_index=False
    )
    .agg(
        unusual_count=(
            "environmental_status",
            lambda x: (x == "Unusual").sum()
        ),
        observations=("environmental_status", "count")
    )
)

india_map = folium.Map(
    location=[22.5, 79.0],
    zoom_start=5,
    tiles="CartoDB positron"
)

for _, row in map_data.iterrows():

    color = (
        "red" if row["unusual_count"] > 0 else "green"
    )

    popup = f"""
    <b>{row['location_id']}</b><br>
    Administrative unit: {row['admin_unit']}<br>
    Observations: {row['observations']}<br>
    Unusual observations: {row['unusual_count']}
    """

    folium.CircleMarker(
        location=[
            row["latitude"],
            row["longitude"]
        ],
        radius=7,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        popup=folium.Popup(popup, max_width=300)
    ).add_to(india_map)

st_folium(
    india_map,
    width=None,
    height=500,
    returned_objects=[]
)

st.caption(
    "Red: location has at least one unusual observation in the "
    "selected period. Green: no unusual observations. "
    "Markers represent 30 sampled cropland locations, not "
    "complete geographic coverage of India."
)

st.divider()

# ==========================================
# VEGETATION HEALTH
# ==========================================

st.subheader("Vegetation Health Trends")

monthly_ndvi = (
    filtered.groupby(
        ["year", "month_number", "month_name"],
        as_index=False
    )["mean_ndvi"]
    .mean()
    .sort_values(["year", "month_number"])
)

fig_ndvi = px.line(
    monthly_ndvi,
    x="month_name",
    y="mean_ndvi",
    color="year",
    markers=True,
    title="Average NDVI Across Available Sample Locations",
    labels={
        "month_name": "Month",
        "mean_ndvi": "Mean NDVI",
        "year": "Year"
    }
)

st.plotly_chart(fig_ndvi, use_container_width=True)

# ==========================================
# RAINFALL TRENDS
# ==========================================

st.subheader("Rainfall Analysis")

monthly_rain = (
    filtered.groupby(
        ["year", "month_number", "month_name"],
        as_index=False
    )["total_rainfall"]
    .mean()
    .sort_values(["year", "month_number"])
)

fig_rain = px.bar(
    monthly_rain,
    x="month_name",
    y="total_rainfall",
    color="year",
    barmode="group",
    title="Average Monthly Rainfall Across Sample Locations",
    labels={
        "month_name": "Month",
        "total_rainfall": "Average Rainfall (mm)"
    }
)

st.plotly_chart(fig_rain, use_container_width=True)

# ==========================================
# ADMINISTRATIVE UNIT COMPARISON
# ==========================================

st.subheader("Administrative Unit Comparison")

admin_summary = (
    filtered.groupby("admin_unit")
    .agg(
        observations=("environmental_status", "count"),
        unusual_count=(
            "environmental_status",
            lambda x: (x == "Unusual").sum()
        ),
        average_ndvi=("mean_ndvi", "mean"),
        average_rainfall=("total_rainfall", "mean")
    )
    .reset_index()
)

fig_admin = px.bar(
    admin_summary.sort_values("unusual_count", ascending=False),
    x="admin_unit",
    y="unusual_count",
    color="unusual_count",
    title="Unusual Observations by Administrative Unit",
    labels={
        "admin_unit": "Administrative Unit",
        "unusual_count": "Unusual Observations"
    }
)

st.plotly_chart(fig_admin, use_container_width=True)

st.dataframe(
    admin_summary.round(3),
    use_container_width=True,
    hide_index=True
)

# ==========================================
# NDVI VS RAINFALL
# ==========================================

st.subheader("Vegetation Health vs Rainfall")

fig_scatter = px.scatter(
    filtered,
    x="total_rainfall",
    y="mean_ndvi",
    color="environmental_status",
    hover_data=[
        "location_id",
        "admin_unit",
        "year",
        "month_name"
    ],
    title="NDVI vs Monthly Rainfall",
    labels={
        "total_rainfall": "Rainfall (mm)",
        "mean_ndvi": "NDVI"
    }
)

st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# UNUSUAL OBSERVATIONS
# ==========================================

st.subheader("Unusual Environmental Observations")

unusual_df = filtered[
    filtered["environmental_status"] == "Unusual"
].sort_values("anomaly_score")

st.dataframe(
    unusual_df[
        [
            "location_id",
            "admin_unit",
            "year",
            "month_name",
            "mean_ndvi",
            "mean_evi",
            "mean_ndmi",
            "total_rainfall",
            "average_temperature",
            "anomaly_score"
        ]
    ].round(3),
    use_container_width=True,
    hide_index=True
)

# ==========================================
# DOWNLOAD DATA
# ==========================================

st.subheader("Download Results")

csv = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Results CSV",
    data=csv,
    file_name="india_crop_health_anomaly_results.csv",
    mime="text/csv"
)

# ==========================================
# PROJECT LIMITATIONS
# ==========================================

st.divider()

st.subheader("Methodology & Limitations")

st.markdown(
    """
    - Satellite data: Sentinel-2 Surface Reflectance Harmonized.
    - Vegetation indices: NDVI, EVI and NDMI.
    - Weather data: NASA POWER daily estimates aggregated monthly.
    - Model: Isolation Forest (unsupervised anomaly detection).
    - Sampling: 30 representative cropland sample points across
      12 administrative units.
    - Period: June–October, 2023–2025.
    - Missing satellite observations are excluded from model training.
    - Anomaly detection identifies unusual environmental combinations,
      not confirmed drought events.
    - This prototype does not provide wall-to-wall coverage of India
      or validated drought predictions.
    """
)

st.success("India-wide monitoring dashboard loaded successfully.")
