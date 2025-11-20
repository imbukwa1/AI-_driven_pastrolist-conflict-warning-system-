# streamlit1.py
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import pydeck as pdk

# ------------------------------------------------------------
# Page
# ------------------------------------------------------------
st.set_page_config(page_title="Integrated Early Warning Dashboard", layout="wide")
st.title("🌍 Integrated Resource Stress & Mobility Anomaly Dashboard")

st.markdown("""
This dashboard combines:

🔸 **NDVI–Rainfall ML Hotspot Model**  
🔸 **CDR Mobility Anomaly Model (Isolation Forest)**  

to give a **compound risk early-warning signal** for Wajir County.
""")

# ------------------------------------------------------------
# Sidebar – county selector
# ------------------------------------------------------------
COUNTIES = ["Wajir"]
county = st.sidebar.selectbox("County", COUNTIES, index=0)

# ------------------------------------------------------------
# Load data function
# ------------------------------------------------------------
@st.cache_data
def load_data(county_name: str):
    base = "../data/processed"
    key = county_name.lower()

    # Enhanced dataset (hotspot + anomaly)
    df = pd.read_csv(f"{base}/{key}_hotspot_enhanced.csv")

    # GeoPoints (for map)
    gdf = gpd.read_file(f"{base}/{key}_hotspot_forecast.geojson")

    # County boundary (optional)
    try:
        boundary = gpd.read_file(f"{base}/{key}_boundary.geojson").to_crs(4326)
        boundary = boundary.dissolve().reset_index(drop=True)
    except:
        boundary = None

    return df, gdf, boundary


df, gdf_geo, boundary = load_data(county)

# ------------------------------------------------------------
# Merge hotspot + geo coordinates
# ------------------------------------------------------------
gdf = gdf_geo.to_crs(4326).copy()
gdf["lon"] = gdf.geometry.x
gdf["lat"] = gdf.geometry.y

df = df.merge(gdf[["cell_id", "lon", "lat"]], on="cell_id", how="left")

# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.header("Map Controls")

hotspot_threshold = st.sidebar.slider("Hotspot threshold", 0.1, 0.9, 0.6, 0.05)
anomaly_threshold = st.sidebar.slider("Mobility anomaly threshold", 0.1, 0.9, 0.5, 0.05)

layer_mode = st.sidebar.selectbox(
    "View mode",
    ["Hotspot only", "Mobility anomaly only", "Combined risk"]
)

# ------------------------------------------------------------
# Prepare filtered data
# ------------------------------------------------------------

df_view = df.copy()

if layer_mode == "Hotspot only":
    df_view = df_view[df_view["p_hotspot_next"] >= hotspot_threshold]

elif layer_mode == "Mobility anomaly only":
    df_view = df_view[df_view["anomaly_prob_norm"] >= anomaly_threshold]

elif layer_mode == "Combined risk":
    df_view = df_view[
        (df_view["p_hotspot_next"] >= hotspot_threshold) &
        (df_view["anomaly_prob_norm"] >= anomaly_threshold)
    ]

# ------------------------------------------------------------
# Map display
# ------------------------------------------------------------

st.subheader("🗺️ Interactive Map")

# Color logic
if layer_mode == "Hotspot only":
    weight_field = "p_hotspot_next"
elif layer_mode == "Mobility anomaly only":
    weight_field = "anomaly_prob_norm"
else:
    # multiply both for compound risk
    df_view["compound_risk"] = df_view["p_hotspot_next"] * df_view["anomaly_prob_norm"]
    weight_field = "compound_risk"

center_lat = df["lat"].mean()
center_lon = df["lon"].mean()

layer = pdk.Layer(
    "HeatmapLayer",
    data=df_view,
    get_position=["lon", "lat"],
    get_weight=weight_field,
    radiusPixels=45,
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=7,
        pitch=0,
    ),
    tooltip={
        "text": "Cell: {cell_id}\nHotspot: {p_hotspot_next}\nMobility anomaly: {anomaly_prob_norm}"
    }
)

st.pydeck_chart(deck)

# ------------------------------------------------------------
# Table
# ------------------------------------------------------------
st.subheader("📄 Filtered Risk Table")

display_cols = [
    "cell_id", "mid_date", "p_hotspot_next",
    "anomaly_prob_norm"
]

if "compound_risk" in df_view:
    display_cols.append("compound_risk")

st.dataframe(df_view[display_cols].sort_values("p_hotspot_next", ascending=False).head(30))

# ------------------------------------------------------------
# Downloads
# ------------------------------------------------------------
st.download_button(
    "⬇️ Download filtered dataset",
    df_view.to_csv(index=False),
    f"{county}_filtered_combined_risk.csv",
    mime="text/csv",
)

# ------------------------------------------------------------
# Notes
# ------------------------------------------------------------
st.markdown("""
### 📘 Model Summary

**Hotspot model:**  
- NDVI 16-day composites  
- Rainfall (CHIRPS 16-day accumulation)  
- NDVI anomaly z-scores  
- Logistic regression / XGBoost baseline  

**Mobility anomaly model (Isolation Forest):**  
- movement_count  
- rolling 7-day and 30-day mobility patterns  
- seasonal movement ratio  
- anomaly probability (0–1)

**Why combine?**  
Hotspot + mobility anomalies = stronger signal for:
- resource pressure  
- herd displacement  
- early conflict triggers  

This supports early interventions (water trucking, offtake, fodder deployment).
""")