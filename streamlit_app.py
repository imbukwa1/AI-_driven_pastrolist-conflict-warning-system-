# streamlit_app.py
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import pydeck as pdk

# -----------------------------
# Page + title
# -----------------------------
st.set_page_config(page_title="Pastoralist Conflict Early Warning", layout="wide")
st.title("🌍 Pastoralist Resource Stress & Conflict Early-Warning")

# -----------------------------
# Sidebar: county picker
#   Add more counties here once you export their files
#   ( <county>_hotspot_forecast.geojson/csv and <county>_boundary.geojson )
# -----------------------------
COUNTIES = [ "Wajir"]  # e.g., ["Turkana", "Marsabit", "Wajir", "Mandera", "Isiolo"]
county = st.sidebar.selectbox("County", COUNTIES, index=0)

# -----------------------------
# Data loaders
# -----------------------------
@st.cache_data
def load_data(county_name: str):
    base = "../data/processed"
    key = county_name.lower()

    # Predictions (points) + table
    gdf_points = gpd.read_file(f"{base}/{key}_hotspot_forecast.geojson")
    df_table   = pd.read_csv(f"{base}/{key}_hotspot_forecast.csv")

    # Boundary polygon (optional but recommended)
    boundary = None
    try:
        boundary = gpd.read_file(f"{base}/{key}_boundary.geojson").to_crs(4326)
        boundary = boundary.dissolve().reset_index(drop=True)  # unify multiparts
    except Exception:
        pass

    return gdf_points, df_table, boundary

gdf, df, boundary = load_data(county)

# -----------------------------
# Sidebar: controls
# -----------------------------
threshold = st.sidebar.slider("Alert Threshold (Probability)", 0.1, 0.9, 0.6, 0.05)
mode = st.sidebar.selectbox("Map style", ["Heatmap (smooth)", "Grid cells (squares)"])

# -----------------------------
# Prep geodata
# -----------------------------
gdf = gdf.to_crs(4326).copy()
gdf["lon"] = gdf.geometry.x
gdf["lat"] = gdf.geometry.y

# Keep only points inside boundary (if boundary available)
if boundary is not None and len(boundary):
    gdf = gpd.sjoin(gdf, boundary, predicate="within", how="inner").drop(columns=["index_right"])

# Filter by alert threshold
gdf_view = gdf[gdf["p_hotspot_next"] >= threshold].copy()

# Center view on boundary (preferred), else points, else default
if boundary is not None and len(boundary):
    minx, miny, maxx, maxy = boundary.total_bounds
elif len(gdf):
    minx, miny, maxx, maxy = gdf.total_bounds
else:
    minx, miny, maxx, maxy = 35.0, 2.0, 37.0, 5.0
center_lat = float((miny + maxy)/2)
center_lon = float((minx + maxx)/2)

# -----------------------------
# Map layers
# -----------------------------
layers = []

if mode.startswith("Heatmap"):
    heat = pdk.Layer(
        "HeatmapLayer",
        data=gdf_view,
        get_position=["lon", "lat"],
        get_weight="p_hotspot_next",
        radiusPixels=45,              # tighten to reduce bleed beyond boundary
        intensity=1.0,
        threshold=0.02,
        colorRange=[
            [150,150,150,160], [210,210,120,190],
            [255,215,0,220], [255,127,0,235], [220,50,47,255],
        ],
    )
    layers.append(heat)
else:
    grid = pdk.Layer(
        "GridCellLayer",
        data=gdf_view,
        get_position=["lon", "lat"],
        cellSize=10000,               # meters (≈10 km cells). Try 5000 for finer.
        elevationScale=0,
        pickable=True,
        get_fill_color="""
        [
            p_hotspot_next >= 0.80 ? 220 :
            p_hotspot_next >= 0.60 ? 255 :
            p_hotspot_next >= 0.40 ? 255 : 150,

            p_hotspot_next >= 0.80 ? 50  :
            p_hotspot_next >= 0.60 ? 127 :
            p_hotspot_next >= 0.40 ? 215 : 150,

            p_hotspot_next >= 0.80 ? 47  :
            p_hotspot_next >= 0.60 ? 0   :
            p_hotspot_next >= 0.40 ? 0   : 150,

            200
        ]
        """,
    )
    layers.append(grid)

# Outline boundary on top for a crisp mask
if boundary is not None and len(boundary):
    boundary_lonlat = boundary.explode(index_parts=False).reset_index(drop=True)
    poly_data = []
    for geom in boundary_lonlat.geometry:
        if geom.geom_type == "Polygon":
            poly_data.append({"polygon": np.array(geom.exterior.coords).tolist()})
        elif geom.geom_type == "MultiPolygon":
            for p in geom.geoms:
                poly_data.append({"polygon": np.array(p.exterior.coords).tolist()})
    poly_layer = pdk.Layer(
        "PolygonLayer",
        data=poly_data,
        get_polygon="polygon",
        get_fill_color=[0,0,0,0],
        get_line_color=[180,180,180],
        lineWidthMinPixels=2,
    )
    layers.append(poly_layer)

view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=7.2, pitch=0)
tooltip = {
    "html": "<b>cell_id:</b> {cell_id}<br/><b>p_hotspot_next:</b> {p_hotspot_next:.3f}",
    "style": {"backgroundColor":"rgba(20,20,20,0.9)", "color":"white"}
}

st.subheader(f"{county}: Cells flagged ≥ {threshold:.2f}: **{(gdf['p_hotspot_next'] >= threshold).sum()}**")
st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip))

# Legend
st.markdown(
    """
    <div style="display:flex;gap:16px;align-items:center;margin:6px 0 18px;">
      <div style="width:14px;height:14px;background:#969696;border-radius:3px;"></div><span>&lt; 0.40 Low</span>
      <div style="width:14px;height:14px;background:#FFD700;border-radius:3px;"></div><span>0.40–0.59 Moderate</span>
      <div style="width:14px;height:14px;background:#FF7F00;border-radius:3px;"></div><span>0.60–0.79 High</span>
      <div style="width:14px;height:14px;background:#DC3230;border-radius:3px;"></div><span>≥ 0.80 Very High</span>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Table + download
# -----------------------------
st.write("📄 Predicted hotspot cells (top 20 by risk):")
st.dataframe(df.sort_values("p_hotspot_next", ascending=False).head(20))

st.download_button(
    "⬇️ Download Forecast CSV",
    df.to_csv(index=False),
    f"{county.lower()}_hotspot_forecast.csv"
)

# -----------------------------
# Notes
# -----------------------------
st.markdown(f"""
### 📘 Method Summary
This model predicts **next-period vegetation stress** for **{county}** using:
- NDVI (vegetation greenness)
- 16-day rainfall accumulation (CHIRPS)
- NDVI anomaly z-scores (seasonal climatology)
- Per-cell lag features

**Hotspots = cells where future NDVI anomaly &lt; −1**  
This is used as a proxy for *pasture stress → resource pressure → potential conflict triggers*.

### ✅ Data Sources
- MODIS NDVI (NASA)
- CHIRPS Rainfall (UC Santa Barbara)
- Google Earth Engine preprocessing
- Logistic Regression / XGBoost baseline

### ⚠️ Ethical Note
This is a **risk-flagging prototype**, not a final peace-action system. No personal data used.
""")