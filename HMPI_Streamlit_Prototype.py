import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import io

st.set_page_config(page_title="HMPI Calculator", layout="wide")

SAMPLE_CSV = """Site, Latitude, Longitude, Pb, Cd, Cr, Ni, Zn, Cu
Site A, 21.1458, 79.0882, 0.015, 0.002, 0.03, 0.01, 1.2, 0.3
Site B, 19.0760, 72.8777, 0.005, 0.001, 0.07, 0.02, 3.4, 0.8
Site C, 28.7041, 77.1025, 0.02, 0.004, 0.1, 0.05, 4.0, 1.5
Site D, 13.0827, 80.2707, 0.008, 0.0005, 0.02, 0.015, 0.6, 0.2
Site E, 22.5726, 88.3639, 0.03, 0.006, 0.2, 0.04, 6.0, 2.0
"""

# Example/default permissible limits (illustrative) 
DEFAULT_LIMITS = {
    "Pb": 0.01,
    "Cd": 0.003,
    "Cr": 0.05,
    "Ni": 0.02,
    "Zn": 5.0,
    "Cu": 2.0
}

def load_sample_df():
    return pd.read_csv(io.StringIO(SAMPLE_CSV))

def read_input(file):
    if file is None:
        return None
    try:
        if hasattr(file, "read"):
            return pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        else:
            return pd.read_csv(file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None

def compute_hpi(df, metals, standards):
    use_metals = [m for m in metals if standards.get(m, 0) > 0]
    if len(use_metals) == 0:
        raise ValueError("No metals with valid (>0) permissible limits.")
    S = np.array([standards[m] for m in use_metals], dtype=float)
    weights = 1.0 / S  # Wi
    M = df[use_metals].fillna(0).to_numpy(dtype=float)
    Qi = (M / S) * 100.0
    contributions = Qi * weights
    numerator = contributions.sum(axis=1)
    denominator = weights.sum()
    hpi = numerator / denominator
    contrib_df = pd.DataFrame(contributions, columns=[f"{m}_contrib" for m in use_metals], index=df.index)
    return pd.Series(hpi, index=df.index), contrib_df, use_metals

def category_to_color(cat):
    if cat == "Low":
        return [30, 160, 60]       # green
    elif cat == "Moderate":
        return [255, 165, 0]       # orange
    else:
        return [200, 30, 30]       # red

# UI
st.title("🚰 HMPI — Heavy Metal Pollution Index Calculator")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV or Excel (columns: site, lat, lon, metal columns)", type=["csv","xlsx","xls"])
    st.markdown("Or download a sample dataset to try:")
    st.download_button("Download sample CSV", SAMPLE_CSV, file_name="hmpi_sample.csv", mime="text/csv")

df = None
if uploaded:
    df = read_input(uploaded)
else:
    if st.button("Use sample dataset"):
        df = load_sample_df()

if df is not None:
    st.subheader("Preview data")
    st.dataframe(df.head())

    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.sidebar.header("Map & Columns")
    site_col = st.sidebar.selectbox("Site name column", options=["None"]+cols, index=0)
    lat_col = st.sidebar.selectbox("Latitude column", options=["None"]+cols, index=cols.index("Latitude") if "Latitude" in cols else 0)
    lon_col = st.sidebar.selectbox("Longitude column", options=["None"]+cols, index=cols.index("Longitude") if "Longitude" in cols else 0)

    st.sidebar.header("Select metal columns")
    known_metals = list(DEFAULT_LIMITS.keys())
    default_mets = [c for c in cols if c in known_metals]
    metals = st.sidebar.multiselect("Metal concentration columns (mg/L)", options=[c for c in cols if c not in [lat_col, lon_col, site_col]], default=default_mets)

    if len(metals) == 0:
        st.warning("Select at least one metal column to compute HPI.")
    else:
        st.sidebar.header("Permissible limits (Si) — mg/L")
        st.sidebar.markdown("Defaults are illustrative. Please verify with official standards.")
        standards = {}
        autofill = st.sidebar.checkbox("Auto-fill example limits (demo only)", value=True)
        for m in metals:
            default = DEFAULT_LIMITS.get(m, 1.0) if autofill else 1.0
            val = st.sidebar.number_input(f"Limit for {m}", min_value=0.0, value=float(default), format="%.6f")
            standards[m] = val

        st.sidebar.header("HPI categories (you can edit)")
        t1 = st.sidebar.number_input("Low/Moderate threshold", min_value=0.0, value=50.0, step=1.0)
        t2 = st.sidebar.number_input("Moderate/High threshold", min_value=0.0, value=100.0, step=1.0)

        if st.button("Compute HMPI"):
            if (lat_col == "None") or (lon_col == "None"):
                st.error("Select latitude and longitude columns to enable map and compute.")
            else:
                try:
                    hpi_series, contrib_df, used_metals = compute_hpi(df, metals, standards)
                except Exception as e:
                    st.error(f"Computation error: {e}")
                else:
                    df_results = df.copy()
                    df_results["HPI"] = np.round(hpi_series, 3)
                    cats = []
                    for h in df_results["HPI"]:
                        if h < t1:
                            cats.append("Low")
                        elif h < t2:
                            cats.append("Moderate")
                        else:
                            cats.append("High")
                    df_results["Category"] = cats
                    df_results = pd.concat([df_results, contrib_df], axis=1)

                    st.success("HMPI computed ✅")
                    st.subheader("Results")
                    st.dataframe(df_results)

                    csv = df_results.to_csv(index=False)
                    st.download_button("Download results CSV", csv, file_name="hmpi_results.csv", mime="text/csv")

                    # Map
                    try:
                        map_df = df_results[[lat_col, lon_col]].copy()
                        map_df = map_df.rename(columns={lat_col: "lat", lon_col: "lon"})

                        # Convert lat/lon to numeric to avoid string errors
                        map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
                        map_df["lon"] = pd.to_numeric(map_df["lon"], errors="coerce")
                        map_df = map_df.dropna(subset=["lat", "lon"])

                        map_df["hpi"] = df_results["HPI"]
                        map_df["site"] = df_results[site_col] if site_col != "None" else df_results.index.astype(str)
                        map_df["category"] = df_results["Category"]
                        map_df["color"] = map_df["category"].apply(category_to_color)
                        map_df["radius"] = (map_df["hpi"] + 1) * 500

                        if not map_df.empty:
                            initial_view = pdk.ViewState(latitude=map_df["lat"].mean(), longitude=map_df["lon"].mean(), zoom=6)
                            layer = pdk.Layer(
                                "ScatterplotLayer",
                                data=map_df,
                                get_position='[lon, lat]',
                                get_color='color',
                                get_radius="radius",
                                pickable=True,
                                auto_highlight=True
                            )
                            r = pdk.Deck(
                                layers=[layer],
                                initial_view_state=initial_view,
                                tooltip={"text": "{site}\nHPI: {hpi}\nCategory: {category}"}
                            )
                            st.subheader("Map — HMPI hotspots")
                            st.pydeck_chart(r)
                        else:
                            st.error("No valid Latitude/Longitude values found in dataset.")

                    except Exception as e:
                        st.error(f"Failed to draw map: {e}")

                    # Metals contribution chart
                    try:
                        contrib_cols = [f"{m}_contrib" for m in used_metals]
                        avg_contrib = df_results[contrib_cols].mean().rename(index=lambda s: s.replace("_contrib",""))
                        st.subheader("Average per-metal contribution to HPI")
                        st.bar_chart(avg_contrib)
                    except Exception as e:
                        st.warning(f"Could not compute contributions chart: {e}")

st.markdown("---")
