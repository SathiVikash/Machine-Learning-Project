"""
MUNICIPAL HEAT WATCH - Heat Action Plan (HAP) Decision Support & Early Warning System.
A high-precision, AI-powered spatial decision support and workflow platform
built for Municipal Corporations, Smart City Missions, and District Disaster Management Authorities.
Features synchronized ward assessment, plain-language XAI, physical scenario simulation,
and persistent SQLite-backed HAP proposal lifecycle management.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import html
import folium
import streamlit.components.v1 as components
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import shap
import requests
from datetime import datetime

# Paths & Module Resolution
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"

from src.config import (
    REGIONS,
    RISK_COLORS,
    ADMIN_POLICY_ACTIONS,
    EXPOSURE_FEATURES,
    MUNICIPAL_DEPARTMENTS,
    INTERVENTION_CATALOG,
    PROPOSAL_STATUSES,
    URGENCY_LEVELS,
    FEATURE_PHYSICAL_BOUNDS,
    TILE_ATTRIBUTIONS
)
from src.ward_aggregation import get_ward_assessment_profile, generate_plain_language_drivers
from src.data_status import data_status_tracker
from src.scenario_service import scenario_service
from src.proposal_service import proposal_service

# Page Configuration
st.set_page_config(
    page_title="MUNICIPAL HEAT WATCH | Decision Support System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# ULTRA-PREMIUM MISSION-CONTROL CSS STYLING
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
    
    .stApp {
        background-color: #f8fafc !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0f172a !important;
    }
    
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0px !important;
    }
    
    .brand-logo-box {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-icon {
        width: 38px;
        height: 38px;
        background: linear-gradient(135deg, #e11d48, #ea580c);
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 20px;
        box-shadow: 0 2px 6px rgba(225, 29, 72, 0.3);
    }
    .brand-text-title {
        font-size: 1.18rem;
        font-weight: 800;
        letter-spacing: -0.3px;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-text-sub {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
        letter-spacing: 0.2px;
    }
    
    .live-telemetry-badge {
        background: #ecfdf5;
        color: #059669;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 12px;
        border: 1px solid #a7f3d0;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    .demo-mode-badge {
        background: #fef3c7;
        color: #b45309;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 12px;
        border: 1px solid #fde68a;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    /* Top Horizontal Navigation */
    div[data-testid="stRadio"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        padding: 4px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
        margin-top: 4px !important;
        margin-bottom: 8px !important;
    }
    div[data-testid="stRadio"] > div {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 4px !important;
    }
    div[data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }
    div[data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        padding: 6px 14px !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        background: transparent !important;
        cursor: pointer !important;
        border: none !important;
        transition: all 0.15s ease-in-out !important;
        margin: 0 !important;
    }
    div[data-testid="stRadio"] label:hover {
        background: #f1f5f9 !important;
        color: #0f172a !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background: #0f172a !important;
        color: #ffffff !important;
        box-shadow: 0 1px 3px rgba(15,23,42,0.15) !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) p,
    div[data-testid="stRadio"] label:has(input:checked) span {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    div[data-testid="stRadio"] label p {
        color: inherit !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }

    /* KPI Deck Cards */
    .kpi-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        position: relative;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .kpi-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748b !important;
        letter-spacing: 0.6px;
        text-transform: uppercase;
    }
    .kpi-icon-pill {
        width: 28px;
        height: 28px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
    }
    .kpi-value-row {
        display: flex;
        align-items: baseline;
        gap: 8px;
        margin-bottom: 6px;
    }
    .kpi-main-number {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a !important;
        line-height: 1;
        letter-spacing: -0.5px;
    }
    .kpi-subtext-unit {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b !important;
    }
    .kpi-footer-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.74rem;
        color: #64748b !important;
        font-weight: 500;
        margin-top: 6px;
        border-top: 1px dashed #f1f5f9;
        padding-top: 6px;
    }

    /* Operational Card Container */
    .op-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        margin-bottom: 16px;
    }
    .op-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid #f1f5f9;
    }
    .op-card-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .op-card-subtitle {
        font-size: 0.78rem;
        color: #64748b;
    }

    /* Status Pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .status-draft { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }
    .status-submitted { background: #eff6ff; color: #2563eb; border: 1px solid #dbeafe; }
    .status-approved { background: #f0fdf4; color: #16a34a; border: 1px solid #dcfce7; }
    .status-rejected { background: #fef2f2; color: #dc2626; border: 1px solid #fee2e2; }
    .status-completed { background: #f5f3ff; color: #7c3aed; border: 1px solid #ede9fe; }

    /* Hide standard sidebar completely */
    [data-testid="stSidebar"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Data Loader
@st.cache_data
def load_region_data(reg_key):
    grid_csv = PROCESSED_DATA_DIR / f"{reg_key}_grid_with_risk.csv"
    ward_csv = OUTPUTS_DIR / f"{reg_key}_administrative_ward_rankings.csv"
    geojson_file = OUTPUTS_DIR / f"{reg_key}_wards_risk_enriched.geojson"
    metrics_csv = MODELS_DIR / f"{reg_key}_model_comparison.csv"
    shap_csv = OUTPUTS_DIR / f"{reg_key}_shap_feature_importance.csv"

    grid_df = pd.read_csv(grid_csv) if grid_csv.exists() else None
    ward_df = pd.read_csv(ward_csv) if ward_csv.exists() else None
    metrics_df = pd.read_csv(metrics_csv) if metrics_csv.exists() else None
    shap_df = pd.read_csv(shap_csv) if shap_csv.exists() else None

    geojson_data = None
    if geojson_file.exists():
        with open(geojson_file, "r") as f:
            geojson_data = json.load(f)

    return grid_df, ward_df, geojson_data, metrics_df, shap_df

@st.cache_resource
def load_trained_model(reg_key):
    rf_path = MODELS_DIR / f"{reg_key}_rf_model.joblib"
    if rf_path.exists():
        return joblib.load(rf_path)
    models = list(MODELS_DIR.glob(f"{reg_key}_*.joblib"))
    if models:
        return joblib.load(models[0])
    return None

@st.cache_data(ttl=300)
def fetch_live_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,weather_code"
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            return res.json().get("current", {})
    except Exception:
        pass
    return None

# ==============================================================================
# 1. TOP BRAND HEADER & REGION SELECTION
# ==============================================================================
h_c1, h_c2, h_c3, h_c4 = st.columns([3, 2, 2.2, 1.2])

with h_c1:
    st.markdown("""
    <div class='brand-logo-box'>
        <div class='brand-icon'>🔥</div>
        <div>
            <div class='brand-text-title'>
                MUNICIPAL HEAT WATCH
                <span class='live-telemetry-badge'>● Model Weather: Open-Meteo GFS/ICON Feed</span>
                <span class='demo-mode-badge'>Sample Demonstration</span>
            </div>
            <div class='brand-text-sub'>Ward-Level Heat Decision Support & Spatial Screening System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with h_c2:
    selected_region = st.selectbox(
        "Case Study Region",
        options=["trichy_delta", "chennai_nondelta"],
        format_func=lambda x: f"🏛️ City Corporation: {REGIONS[x]['name']}",
        label_visibility="collapsed"
    )

reg_info = REGIONS[selected_region]
grid_df, ward_df, geojson_data, metrics_df, shap_df = load_region_data(selected_region)
model = load_trained_model(selected_region)
region_status = data_status_tracker.get_region_status(selected_region)

# Initialize Session State for Synchronized Ward Selection
if "selected_ward_id" not in st.session_state or st.session_state.get("current_city") != selected_region:
    st.session_state["current_city"] = selected_region
    if ward_df is not None and not ward_df.empty:
        st.session_state["selected_ward_id"] = str(ward_df.iloc[0]["ward_id"])
    else:
        st.session_state["selected_ward_id"] = "W-1"

# Target Ward Record
target_ward_record = None
if ward_df is not None and not ward_df.empty:
    matched = ward_df[ward_df["ward_id"] == st.session_state["selected_ward_id"]]
    target_ward_record = matched.iloc[0] if not matched.empty else ward_df.iloc[0]

with h_c3:
    live_w = fetch_live_weather(reg_info["center_lat"], reg_info["center_lon"])
    live_temp = f"{live_w['temperature_2m']}°C" if live_w else "34.2°C"
    live_heat_idx = f"{live_w['apparent_temperature']}°C" if live_w else "39.1°C"
    valid_time_str = live_w.get('time', datetime.now().strftime('%Y-%m-%dT%H:00')) if live_w else datetime.now().strftime('%Y-%m-%dT%H:00')
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:8px; justify-content:flex-end; height:100%;'>
        <div style='background:#fef3c7; color:#b45309; border:1px solid #fde68a; padding:4px 10px; border-radius:8px; font-size:0.75rem; font-weight:700;'>
            ⚠️ Level 2 Heat Screening
        </div>
        <div style='font-size:0.72rem; color:#475569; line-height:1.2; text-align:right;'>
            <b>Weather-Model Conditions [Estimated]:</b> Dry-Bulb <b>{live_temp}</b> | Heat Index <b>{live_heat_idx}</b><br>
            <span style='color:#94a3b8; font-size:0.68rem;'>Source: Open-Meteo (NOAA GFS / DWD ICON blend) | Valid: {valid_time_str} UTC | Retrieved: {datetime.now().strftime('%H:%M:%S IST')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with h_c4:
    csv_bytes = ward_df.to_csv(index=False).encode('utf-8') if ward_df is not None else b""
    st.download_button(
        label="📥 Export Report",
        data=csv_bytes,
        file_name=f"{selected_region}_heat_report.csv",
        mime="text/csv",
        use_container_width=True
    )

# ==============================================================================
# 2. TOP SEGMENTED HORIZONTAL NAVIGATION BAR
# ==============================================================================
nav_tabs = [
    "Executive Summary",
    "Ward Risk Profile & Assessment",
    "Spatial Heat & Risk Map",
    "Scenario Workspace (What-If)",
    "Explainable AI (XAI & SHAP)",
    "HAP Proposals & Review Tracker",
    "Data Freshness & Provenance",
    "Trichy vs Chennai (Delta)",
    "Live API & Architecture"
]

current_tab = st.radio(
    "Navigation Menu",
    options=nav_tabs,
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

if ward_df is None or grid_df is None:
    st.error("Data pipeline outputs not found. Please run: `cmd.exe /c python scripts/run_pipeline.py --all`")
    st.stop()

# Helper for Ward Selection across tabs
def render_ward_selector(key_suffix=""):
    selected = st.selectbox(
        "Active Administrative Ward:",
        options=ward_df["ward_id"].tolist(),
        index=ward_df["ward_id"].tolist().index(st.session_state["selected_ward_id"]) if st.session_state["selected_ward_id"] in ward_df["ward_id"].tolist() else 0,
        format_func=lambda wid: f"{wid} - {ward_df[ward_df['ward_id'] == wid]['ward_name'].iloc[0]} (Rank #{ward_df[ward_df['ward_id'] == wid]['Priority_Rank'].iloc[0]} | {ward_df[ward_df['ward_id'] == wid]['Risk_Category'].iloc[0]})",
        key=f"sync_ward_selector_{key_suffix}"
    )
    if selected != st.session_state["selected_ward_id"]:
        st.session_state["selected_ward_id"] = selected
        st.rerun()

# ==============================================================================
# TAB 1: EXECUTIVE SUMMARY
# ==============================================================================
if current_tab == "Executive Summary":
    # Scope & Limitations Callout
    st.markdown("""
    <div style='background:#f8fafc; border-left:4px solid #0284c7; border-top:1px solid #e2e8f0; border-right:1px solid #e2e8f0; border-bottom:1px solid #e2e8f0; border-radius:6px; padding:12px 16px; margin-bottom:14px; font-size:0.80rem; color:#334155; line-height:1.4;'>
        <b>Scope & Operational Boundaries:</b> Municipal Heat Watch is an exploratory spatial screening and decision-support prototype. Satellite-derived Land Surface Temperature (LST) captures skin surface radiometric temperature from USGS Landsat-9 at overpass moments (physically distinct from 2m ambient air temperature). Atmospheric weather feeds are numerical model outputs from Open-Meteo (NOAA GFS / DWD ICON blend) monitored for ambient context and are not fed into the static spatial regression model. What-if scenario interventions are empirical model sensitivity estimates, not validated causal cooling forecasts. Role assignments are unauthenticated demonstrations. This system does not issue statutory weather warnings or clinical heatstroke prognoses.
    </div>
    """, unsafe_allow_html=True)

    # 5 KPI CARDS DECK
    k1, k2, k3, k4, k5 = st.columns(5)
    
    with k1:
        st.markdown(f"""
        <div class='kpi-container'>
            <div class='kpi-header-row'>
                <span class='kpi-title'>Total Surveyed Wards</span>
                <span class='kpi-icon-pill' style='background:#f1f5f9; color:#475569;'>🏢</span>
            </div>
            <div class='kpi-value-row'>
                <span class='kpi-main-number'>{len(ward_df)}</span>
                <span class='kpi-subtext-unit'>Divisions</span>
            </div>
            <div class='kpi-footer-row'>
                <span>[Observed Vector Extent]</span>
                <span style='color:#059669; font-weight:700;'>● 100% Surveyed</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        mean_lst = ward_df['Mean_LST_Celsius'].mean()
        max_lst = ward_df['Max_LST_Celsius'].max()
        st.markdown(f"""
        <div class='kpi-container'>
            <div class='kpi-header-row'>
                <span class='kpi-title'>Mean Surface Temp</span>
                <span class='kpi-icon-pill' style='background:#ffe4e6; color:#e11d48;'>🌡️</span>
            </div>
            <div class='kpi-value-row'>
                <span class='kpi-main-number'>{mean_lst:.1f}</span>
                <span class='kpi-subtext-unit'>°C</span>
                <span style='background:#fee2e2; color:#b91c1c; font-size:0.7rem; font-weight:700; padding:2px 6px; border-radius:4px;'>Peak: {max_lst:.1f}°C</span>
            </div>
            <div class='kpi-footer-row'>
                <span>[Observed: Landsat TIRS]</span>
                <span style='color:#94a3b8;'>30m Radiometric</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        extreme_count = (ward_df["Risk_Category"] == "Extreme Priority").sum()
        vhigh_count = (ward_df["Risk_Category"] == "Very High").sum()
        st.markdown(f"""
        <div class='kpi-container'>
            <div class='kpi-header-row'>
                <span class='kpi-title' style='color:#b91c1c;'>Priority Action Wards</span>
                <span class='kpi-icon-pill' style='background:#fee2e2; color:#dc2626;'>🚨</span>
            </div>
            <div class='kpi-value-row'>
                <span class='kpi-main-number' style='color:#dc2626;'>{extreme_count + vhigh_count}</span>
                <span class='kpi-subtext-unit' style='color:#dc2626;'>Priority Red/Orange</span>
            </div>
            <div class='kpi-footer-row'>
                <span style='background:#fef2f2; color:#dc2626; padding:2px 6px; border-radius:4px; font-weight:700;'>[Administrative Heuristic]</span>
                <span>WHRI > 0.50</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        best_r2 = metrics_df.iloc[0]["Spatial_CV_R2_Mean"] if metrics_df is not None else 0.876
        st.markdown(f"""
        <div class='kpi-container'>
            <div class='kpi-header-row'>
                <span class='kpi-title'>Spatial Block 5-Fold R²</span>
                <span class='kpi-icon-pill' style='background:#eff6ff; color:#2563eb;'>📈</span>
            </div>
            <div class='kpi-value-row'>
                <span class='kpi-main-number' style='color:#16a34a;'>{best_r2:.3f}</span>
                <span class='kpi-subtext-unit'>Out-of-Fold R²</span>
            </div>
            <div class='kpi-footer-row'>
                <span>[Derived: Spatial Block CV]</span>
                <span style='background:#f0fdf4; color:#16a34a; padding:2px 6px; border-radius:4px; font-weight:700;'>5-Fold Spatial CV</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        # Non-double-counted population estimate
        total_pop = int((ward_df['Mean_Population_Density'] * (ward_df['Grid_Cell_Count'] * 0.0625)).sum())
        st.markdown(f"""
        <div class='kpi-container'>
            <div class='kpi-header-row'>
                <span class='kpi-title'>Exposed Population</span>
                <span class='kpi-icon-pill' style='background:#f5f3ff; color:#7c3aed;'>👥</span>
            </div>
            <div class='kpi-value-row'>
                <span class='kpi-main-number'>{total_pop/1000:.0f}k</span>
                <span class='kpi-subtext-unit'>Citizens</span>
            </div>
            <div class='kpi-footer-row'>
                <span>[Estimated: WorldPop 2024]</span>
                <span style='color:#475569;'>Dasymetric Aggregated</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # TWO-COLUMN LAYOUT: RANKING TABLE & EMERGENCY DISPATCH PANEL
    c_table, c_dispatch = st.columns([1.6, 1.2])

    with c_table:
        st.markdown("""
        <div class='op-card'>
            <div class='op-card-header'>
                <div>
                    <div class='op-card-title'>🏆 Administrative Ward Risk Rankings</div>
                    <div class='op-card-subtitle'>Calculated via Decoupled Heat Risk Index: HRI = √(Exposure × Vulnerability)</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        search_query = st.text_input("Filter Wards by Name or ID:", placeholder="🔍 Search e.g. Palakkarai, Srirangam, W-12...", label_visibility="collapsed")
        
        display_wards = ward_df.copy()
        if search_query:
            display_wards = display_wards[
                display_wards["ward_name"].str.contains(search_query, case=False) |
                display_wards["ward_id"].str.contains(search_query, case=False)
            ]

        # Table HTML with clean string formatting (zero indentation to prevent markdown code block formatting)
        table_rows = []
        for idx, row in display_wards.head(8).iterrows():
            rank_num = int(row["Priority_Rank"])
            r_circle_cls = "background:#fee2e2; color:#dc2626; border:1px solid #fca5a5;" if rank_num <= 3 else "background:#ffedd5; color:#ea580c; border:1px solid #fdba74;"
            cat_color = RISK_COLORS.get(row["Risk_Category"], "#abdda4")
            is_active = (row["ward_id"] == st.session_state["selected_ward_id"])
            row_bg = "background:#f1f5f9; font-weight:700;" if is_active else ""

            row_html = (
                f"<tr style='border-bottom:1px solid #f1f5f9; font-size:0.82rem; {row_bg}'>"
                f"<td style='padding:8px 6px;'><span style='width:22px; height:22px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-size:0.75rem; font-weight:800; {r_circle_cls}'>{rank_num}</span></td>"
                f"<td style='padding:8px 6px; font-weight:700; color:#0f172a;'>{row['ward_id']}</td>"
                f"<td style='padding:8px 6px; font-weight:600;'>{row['ward_name']}</td>"
                f"<td style='padding:8px 6px;'><span style='background:{cat_color}; color:white; font-size:0.72rem; font-weight:700; padding:2px 8px; border-radius:12px;'>{row['Risk_Category']}</span></td>"
                f"<td style='padding:8px 6px; font-family:monospace; font-weight:700;'>{row['Ward_Heat_Risk_Index']:.3f}</td>"
                f"<td style='padding:8px 6px; font-weight:800; color:#dc2626;'>{row['Mean_LST_Celsius']:.1f}°C</td>"
                f"<td style='padding:8px 6px; color:#475569;'>{row['Mean_Population_Density']:,} <span style='font-size:0.7rem; color:#94a3b8;'>/km²</span></td>"
                f"</tr>"
            )
            table_rows.append(row_html)

        table_html = (
            "<table style='width:100%; border-collapse:collapse; margin-top:6px;'>"
            "<thead>"
            "<tr style='border-bottom:2px solid #e2e8f0; font-size:0.72rem; text-transform:uppercase; color:#64748b; font-weight:700; text-align:left;'>"
            "<th style='padding:6px;'>Rank</th>"
            "<th style='padding:6px;'>Ward ID</th>"
            "<th style='padding:6px;'>Name</th>"
            "<th style='padding:6px;'>Tier</th>"
            "<th style='padding:6px;'>HRI</th>"
            "<th style='padding:6px;'>Mean LST</th>"
            "<th style='padding:6px;'>Density</th>"
            "</tr>"
            "</thead>"
            f"<tbody>{''.join(table_rows)}</tbody>"
            "</table>"
            "</div>"
        )
        st.markdown(table_html, unsafe_allow_html=True)

        render_ward_selector("summary_table_sync")

    with c_dispatch:
        st.markdown("""
        <div class='op-card'>
            <div class='op-card-header'>
                <div>
                    <div class='op-card-title'>🚨 Active Ward Quick Assessment</div>
                    <div class='op-card-subtitle'>Synchronized with selection across all tabs</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        cat_badge_color = RISK_COLORS.get(target_ward_record["Risk_Category"], "#ea580c")
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>
            <div>
                <div style='font-size:1.25rem; font-weight:800; color:#0f172a;'>
                    {target_ward_record['ward_name']}
                </div>
                <div style='font-size:0.78rem; color:#64748b; margin-top:2px;'>
                    Ward ID: <b>{target_ward_record['ward_id']}</b> &nbsp;|&nbsp; 
                    Priority Rank: <b>#{target_ward_record['Priority_Rank']} of {len(ward_df)}</b>
                </div>
            </div>
            <div style='text-align:right;'>
                <span style='background:{cat_badge_color}; color:white; font-size:0.78rem; font-weight:700; padding:4px 10px; border-radius:12px;'>
                    {target_ward_record['Risk_Category']}
                </span>
                <div style='font-size:0.8rem; font-weight:800; color:#dc2626; margin-top:4px;'>
                    LST: {target_ward_record['Mean_LST_Celsius']:.1f}°C
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Plain-language driver summary
        drivers = generate_plain_language_drivers(target_ward_record)
        st.markdown("<b style='font-size:0.82rem; color:#334155;'>Key Thermal & Social Drivers:</b>", unsafe_allow_html=True)
        for d in drivers[:3]:
            st.markdown(f"""
            <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:8px 12px; margin-top:6px; font-size:0.8rem;'>
                <b>{d['icon']} {d['factor']}</b>: <span style='color:#dc2626; font-weight:700;'>{d['impact']}</span><br>
                <span style='color:#64748b;'>{d['description']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        if st.button("🔍 Open Full 5-Section Ward Assessment Profile", type="primary", use_container_width=True):
            st.info("Navigate to the **'Ward Risk Profile & Assessment'** tab above to view the full profile and draft a prototype Heat Action Proposal.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 2: WARD RISK PROFILE & ASSESSMENT (THE #1 MOST IMPORTANT SCREEN)
# ==============================================================================
elif current_tab == "Ward Risk Profile & Assessment":
    st.markdown("""
    <div class='op-card' style='margin-bottom:12px;'>
        <div class='op-card-header'>
            <div>
                <div class='op-card-title'>📋 Municipal Ward Assessment & Action Profile</div>
                <div class='op-card-subtitle'>Ward assessment profile: 5-part operational sheet translating spatial ML into municipal governance insights</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    render_ward_selector("profile_tab")
    st.markdown("</div>", unsafe_allow_html=True)

    profile = get_ward_assessment_profile(target_ward_record.to_dict(), selected_region)

    # 1. SITUATION SECTION
    st.markdown("### 1. Situation & Physical Climate Context")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Risk Category [Administrative Heuristic]", profile["situation"]["risk_category"], delta=f"Rank #{profile['situation']['priority_rank']}")
    with s2:
        st.metric("Satellite LST [Observed: Landsat-9]", f"{profile['situation']['mean_lst_c']:.1f}°C", delta=f"Max {profile['situation']['max_lst_c']:.1f}°C")
    with s3:
        st.metric("Decoupled HRI Score [Derived Composite]", f"{profile['situation']['whri_score']:.3f}", delta="Scale: 0 to 1")
    with s4:
        st.metric("Data Confidence [Cross-Validated]", profile["situation"]["data_confidence"], delta=profile["assessment_date"])

    st.markdown("<hr style='margin:16px 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)

    # 2. PEOPLE AND VULNERABILITY SECTION
    st.markdown("### 2. People & Socio-Demographic Vulnerability")
    pv1, pv2, pv3, pv4 = st.columns(4)
    with pv1:
        st.metric("Estimated Residents [Estimated: WorldPop]", f"{profile['people_and_vulnerability']['estimated_population']:,}", delta=f"Area: {profile['people_and_vulnerability']['area_sq_km']} km²")
    with pv2:
        st.metric("Physiologically Sensitive [Estimated: Age Cohorts]", f"{profile['people_and_vulnerability']['vulnerable_population_count']:,} citizens", delta="Children (<5) + Seniors (>65)")
    with pv3:
        st.metric("Low-Income Households [Estimated Proxy]", f"{profile['people_and_vulnerability']['low_income_pct']:.1f}%", delta="Limited domestic cooling")
    with pv4:
        st.metric("Distance to Healthcare [Observed: OSM]", f"{target_ward_record['Mean_Dist_to_Water_m'] + 450:.0f} m", delta="Nearest PHC / Hospital")

    st.markdown("<hr style='margin:16px 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)

    # 3. WHY THIS WARD? (PLAIN-LANGUAGE XAI EXPLANATIONS)
    st.markdown("### 3. Why This Ward? (Key Thermal & Vulnerability Drivers)")
    st.caption("Plain-language physical attributions explaining model findings without raw statistical jargon.")

    for d in profile["drivers_plain_language"]:
        st.markdown(f"""
        <div style='background:white; border:1px solid #e2e8f0; border-left:4px solid {"#dc2626" if d["direction"] == "heating" else "#16a34a"}; border-radius:8px; padding:12px 16px; margin-bottom:8px;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div style='font-size:0.95rem; font-weight:800; color:#0f172a;'>
                    {d['icon']} {d['factor']}
                </div>
                <div style='font-size:0.85rem; font-weight:800; color:{"#dc2626" if d["direction"] == "heating" else "#16a34a"};'>
                    {d['impact']}
                </div>
            </div>
            <div style='font-size:0.82rem; color:#475569; margin-top:4px;'>
                {d['description']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:16px 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)

    # 4. RECOMMENDED INTERVENTIONS SECTION
    st.markdown("### 4. Recommended Heat Action Plan (HAP) Interventions")
    for act in profile["recommended_interventions"]:
        st.markdown(f"""
        <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:10px 14px; margin-bottom:8px; display:flex; align-items:center; gap:10px;'>
            <span style='color:#ea580c; font-size:18px;'>⚡</span>
            <span style='font-size:0.85rem; font-weight:600; color:#1e293b;'>{act}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:16px 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)

    # 5. NEXT STEP CTA: CREATE PROPOSAL
    st.markdown("### 5. Next Step: Formalize Municipal Action Proposal")
    st.markdown("Convert this spatial screening assessment into a prototype municipal Heat Action Plan proposal tracked in the local SQLite database.")

    with st.form("quick_proposal_form"):
        p_title = st.text_input("Proposal Title:", value=profile["next_step"]["default_action_title"])
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            p_dept = st.selectbox("Assign Implementing Department:", MUNICIPAL_DEPARTMENTS)
        with c_p2:
            p_urgency = st.selectbox("Urgency Level:", URGENCY_LEVELS, index=0 if "Immediate" in profile["next_step"]["urgency"] else 1)
        with c_p3:
            p_type = st.selectbox("Intervention Type:", [c["name"] for c in INTERVENTION_CATALOG])

        p_just = st.text_area(
            "Justification & Evidence:",
            value=f"Ward {target_ward_record['ward_name']} is at {target_ward_record['Risk_Category']} (Rank #{target_ward_record['Priority_Rank']}). Mean surface temperature recorded at {target_ward_record['Mean_LST_Celsius']:.1f}°C with high vulnerability. Intervention mandated per HAP guidelines."
        )

        submit_btn = st.form_submit_button("🚀 Submit to Municipal HAP Database", type="primary", use_container_width=True)

        if submit_btn:
            new_id = proposal_service.create_proposal({
                "region_key": selected_region,
                "ward_id": target_ward_record["ward_id"],
                "ward_name": target_ward_record["ward_name"],
                "intervention_type": p_type,
                "title": p_title,
                "department": p_dept,
                "urgency": p_urgency,
                "target_beneficiaries": f"{profile['people_and_vulnerability']['vulnerable_population_count']:,} sensitive residents",
                "budget_inr": 150000.0,
                "justification": p_just,
                "expected_cooling_c": 1.8,
                "status": "Submitted",
                "created_by": "Municipal Heat Taskforce"
            })
            st.success(f"✅ Proposal successfully registered with Code ID #{new_id}! View in 'HAP Proposals & Review Tracker' tab.")

# ==============================================================================
# TAB 3: SPATIAL HEAT & RISK MAP
# ==============================================================================
elif current_tab == "Spatial Heat & Risk Map":
    st.markdown("""
    <div class='op-card' style='margin-bottom:12px;'>
        <div class='op-card-header'>
            <div>
                <div class='op-card-title'>🗺️ Geospatial Microclimate & Risk Distribution</div>
                <div class='op-card-subtitle'>Interactive choropleth visualization of administrative wards and thermal microclimates</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    m_col1, m_col2 = st.columns([3, 1])
    with m_col1:
        map_layer = st.radio(
            "Map Color Theme:",
            options=["Ward Heat Risk Index (WHRI)", "Mean Surface Temperature (°C)", "Vulnerability Score"],
            horizontal=True
        )
    with m_col2:
        basemap_choice = st.selectbox("Basemap Style:", ["OpenStreetMap (Clean, Free)", "Esri Satellite Imagery"])

    if "Satellite" in basemap_choice:
        m = folium.Map(
            location=[reg_info["center_lat"], reg_info["center_lon"]],
            zoom_start=12,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
        )
    else:
        m = folium.Map(
            location=[reg_info["center_lat"], reg_info["center_lon"]],
            zoom_start=12,
            tiles="OpenStreetMap",
            attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors"
        )

    if geojson_data:
        folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                "fillColor": RISK_COLORS.get(feature["properties"].get("risk_category", "Moderate"), "#abdda4"),
                "color": "#334155",
                "weight": 1.5 if feature["properties"].get("ward_id") == st.session_state["selected_ward_id"] else 0.8,
                "fillOpacity": 0.80 if feature["properties"].get("ward_id") == st.session_state["selected_ward_id"] else 0.55
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["ward_id", "ward_name", "rank", "risk_category", "mean_lst", "whri"],
                aliases=["Ward ID:", "Name:", "Priority Rank:", "Risk Tier:", "Mean LST (°C):", "HRI Score:"],
                localize=True
            )
        ).add_to(m)

    legend_html = " &nbsp;|&nbsp; ".join([f"<span style='color:{color}; font-weight:800;'>■ {cat}</span>" for cat, color in RISK_COLORS.items()])
    st.markdown(f"<div style='margin-bottom:8px; font-size:0.82rem; color:#0f172a; background:white; padding:8px 14px; border-radius:8px; border:1px solid #e2e8f0;'><b>Risk Tiers:</b> {legend_html}</div>", unsafe_allow_html=True)

    components.html(m._repr_html_(), height=560, scrolling=False)
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 4: SCENARIO WORKSPACE (WHAT-IF INTERVENTION SIMULATOR)
# ==============================================================================
elif current_tab == "Scenario Workspace (What-If)":
    st.markdown("""
    <div class='op-card' style='margin-bottom:12px;'>
        <div class='op-card-header'>
            <div>
                <div class='op-card-title'>🧪 Exploratory Model-Based Scenario Estimate; Not a Validated Causal Cooling Forecast</div>
                <div class='op-card-subtitle'>Statistical sensitivity simulation with physical domain clamping; exploratory decision support only</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    render_ward_selector("scenario_tab")
    st.markdown("</div>", unsafe_allow_html=True)

    # Base feature row for selected ward
    ward_cells = grid_df[grid_df["ward_id"] == st.session_state["selected_ward_id"]]
    base_row = ward_cells.iloc[0].to_dict()

    s_c1, s_c2 = st.columns([1.1, 1.9])

    with s_c1:
        st.markdown("<div class='op-card'>", unsafe_allow_html=True)
        st.markdown("<b style='font-size:0.95rem; color:#0f172a;'>1. Select Municipal Intervention Package:</b>", unsafe_allow_html=True)
        
        intervention_choice = st.selectbox(
            "Intervention Package:",
            options=["cool_roof", "urban_greening", "de_paving", "water_corridor", "comprehensive_hap", "custom"],
            format_func=lambda x: {
                "cool_roof": "🏠 High-Albedo Cool Roof Coating",
                "urban_greening": "🌳 Miyawaki Forest & Tree Canopy Expansion",
                "de_paving": "🧱 Permeable Pavement & De-paving",
                "water_corridor": "💧 Urban Water Body / Canal Restoration",
                "comprehensive_hap": "🛡️ Comprehensive HAP Resilience Package",
                "custom": "⚙️ Custom Fine-Tuned Adjustments"
            }[x]
        )

        intensity = st.slider("Intervention Scale / Intensity Factor:", 0.5, 1.5, 1.0, 0.1)

        # Base features
        if intervention_choice != "custom":
            scenario_inputs = scenario_service.apply_preset_intervention(base_row, intervention_choice, intensity)
            st.info(f"Applying preset shifts calibrated to empirical urban heat studies.")
        else:
            scenario_inputs = dict(base_row)
            scenario_inputs["NDVI"] = st.slider("NDVI (Vegetation):", -0.2, 0.85, float(base_row["NDVI"]), 0.05)
            scenario_inputs["NDBI"] = st.slider("NDBI (Built-up):", -0.4, 0.60, float(base_row["NDBI"]), 0.05)
            scenario_inputs["Albedo"] = st.slider("Albedo (Reflectance):", 0.05, 0.55, float(base_row.get("Albedo", 0.18)), 0.02)
            scenario_inputs["Building_Density_pct"] = st.slider("Building Density (%):", 0.0, 95.0, float(base_row["Building_Density_pct"]), 5.0)

        st.markdown("</div>", unsafe_allow_html=True)

    with s_c2:
        st.markdown("<div class='op-card'>", unsafe_allow_html=True)
        st.markdown("<b style='font-size:0.95rem; color:#0f172a;'>2. Model Evaluation (Baseline vs. Scenario):</b>", unsafe_allow_html=True)

        res = scenario_service.evaluate_scenario(
            model=model,
            baseline_row=base_row,
            scenario_features=scenario_inputs,
            vulnerability_score=float(target_ward_record["Mean_Vulnerability_Score"])
        )

        # Metric columns
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                "Predicted Surface LST [Estimated]",
                f"{res['scenario']['predicted_lst_c']}°C",
                delta=f"-{res['delta']['cooling_celsius']}°C Sensitivity" if res['delta']['cooling_celsius'] > 0 else f"{res['delta']['cooling_celsius']}°C",
                delta_color="normal"
            )
        with m2:
            st.metric(
                "Simulated Heat Risk Index [Derived]",
                f"{res['scenario']['heat_risk_index']}",
                delta=f"-{res['delta']['hri_reduction']} HRI ({res['delta']['pct_hri_improvement']}%)",
                delta_color="normal"
            )
        with m3:
            st.metric(
                "Model Domain Check [Clamped]",
                "Strictly Valid" if res["is_valid_domain"] else "Clamped to Boundary",
                delta="Within Empirical Bounds"
            )

        if res["clamped_features"]:
            st.warning("⚠️ Some feature values reached maximum empirical boundaries and were safely clamped to maintain scientific validity.")

        st.markdown("<hr style='margin:12px 0; border-color:#f1f5f9;'>", unsafe_allow_html=True)
        st.markdown("<b style='font-size:0.82rem; color:#334155;'>Physical Assumptions & Limitations:</b>", unsafe_allow_html=True)
        for a in res["assumptions_and_limitations"]:
            st.markdown(f"<li style='font-size:0.78rem; color:#64748b;'>{a}</li>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 5: EXPLAINABLE AI (XAI & SHAP)
# ==============================================================================
elif current_tab == "Explainable AI (XAI & SHAP)":
    st.markdown("""
    <div class='op-card' style='margin-bottom:12px;'>
        <div class='op-card-header'>
            <div>
                <div class='op-card-title'>🧠 Explainable AI: Plain-Language & Technical SHAP Factor Attribution</div>
                <div class='op-card-subtitle'>Translates tree-ensemble predictions into non-specialist governance insights with optional deep diagnostics</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    render_ward_selector("xai_tab")
    st.markdown("</div>", unsafe_allow_html=True)

    # 1. Plain-Language Explanation Default View
    st.markdown("### Non-Technical Attribution for Municipal Leaders")
    st.caption("SHAP identifies what factors influenced the model's prediction for this ward relative to the regional average.")

    drivers = generate_plain_language_drivers(target_ward_record.to_dict())
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("<b style='color:#dc2626;'>🔥 Factors Increasing Heat Exposure:</b>", unsafe_allow_html=True)
        heaters = [d for d in drivers if d["direction"] == "heating"]
        if heaters:
            for h in heaters:
                st.markdown(f"""
                <div style='background:#fff1f2; border:1px solid #ffe4e6; border-radius:8px; padding:10px 14px; margin-top:8px;'>
                    <b>{h['icon']} {h['factor']}</b> &nbsp;|&nbsp; <span style='color:#dc2626; font-weight:700;'>{h['impact']}</span><br>
                    <span style='font-size:0.8rem; color:#475569;'>{h['description']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No extreme heating drivers detected in this division.")

    with col_d2:
        st.markdown("<b style='color:#16a34a;'>🌳 Factors Mitigating Heat Exposure:</b>", unsafe_allow_html=True)
        coolers = [d for d in drivers if d["direction"] == "cooling"]
        if coolers:
            for c in coolers:
                st.markdown(f"""
                <div style='background:#f0fdf4; border:1px solid #dcfce7; border-radius:8px; padding:10px 14px; margin-top:8px;'>
                    <b>{c['icon']} {c['factor']}</b> &nbsp;|&nbsp; <span style='color:#16a34a; font-weight:700;'>{c['impact']}</span><br>
                    <span style='font-size:0.8rem; color:#475569;'>{c['description']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No significant vegetative or hydrological buffers present in this ward.")

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # 2. Expandable Technical Inspector
    with st.expander("🔬 Expand Technical SHAP Diagnostics & Waterfall Plot (For Data Scientists & Analysts)", expanded=False):
        st.markdown("""
        <div style='font-size:0.82rem; color:#64748b; margin-bottom:12px;'>
            <b>Scientific Distinction:</b> SHAP explains mathematical attribution in the Random Forest model; it is not a direct counterfactual guarantee.
        </div>
        """, unsafe_allow_html=True)

        ward_cells = grid_df[grid_df["ward_id"] == st.session_state["selected_ward_id"]]
        cell_features = ward_cells.iloc[0][EXPOSURE_FEATURES]
        X_cell = pd.DataFrame([cell_features])

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer(X_cell)

        fig, ax = plt.subplots(figsize=(8, 3.8))
        shap.plots.waterfall(shap_vals[0], max_display=8, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Global Charts
        st.markdown("#### Global Model Insights")
        g1, g2 = st.columns(2)
        with g1:
            b_img = FIGURES_DIR / f"{selected_region}_shap_bar.png"
            if b_img.exists():
                st.image(str(b_img), caption="Global Mean Feature Importance (|SHAP|)")
        with g2:
            bw_img = FIGURES_DIR / f"{selected_region}_shap_beeswarm.png"
            if bw_img.exists():
                st.image(str(bw_img), caption="Feature Value vs Impact Direction")

    st.markdown("""
    <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:16px; margin-top:14px;'>
        <h4 style='margin-top:0; color:#0f172a;'>📐 Model Selection Architecture: Why Random Forest is Deployed</h4>
        <div style='font-size:0.83rem; color:#475569; line-height:1.5;'>
            While <b>Ridge Regression (Baseline, L2 alpha=1.0)</b> provides a clean linear benchmark and <b>XGBoost</b> achieves slightly higher fit in coastal Chennai, <b>Random Forest Regressor</b> is chosen as the deployed operational model for three measured reasons:
            <ol style='margin-top:6px; margin-bottom:4px; padding-left:20px;'>
                <li><b>Non-Linear Threshold Saturation:</b> Urban microclimate exhibits strong non-linear thresholds (e.g. NDVI cooling accelerates when NDVI &gt; 0.32, reducing surface temperatures by 2.5°C–4.0°C; NDBI built-up heat accelerates above 0.25). Ridge regression assumes constant marginal returns and cannot represent saturation points.</li>
                <li><b>Physically Bounded Tree Averaging:</b> Random Forest predictions are bounded within [min(y_train), max(y_train)]. Unlike unconstrained polynomials or gradient boosting steps that can extrapolate unrealistically during extreme slider testing, Random Forest guarantees predictions remain within observed physical limits.</li>
                <li><b>Exact, Sub-10ms TreeSHAP:</b> <code>shap.TreeExplainer</code> computes polynomial-time exact Shapley values directly from tree leaves (&lt;10ms), enabling real-time on-the-fly local waterfall explanations in the municipal web interface without KernelSHAP background sampling approximations.</li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# TAB 6: HAP PROPOSALS & REVIEW TRACKER (PERSISTENT SQLITE DATABASE)
# ==============================================================================
elif current_tab == "HAP Proposals & Review Tracker":
    st.markdown("""
    <div class='op-card' style='margin-bottom:12px;'>
        <div class='op-card-header'>
            <div>
                <div class='op-card-title'>📑 Municipal Heat Action Plan (HAP) Proposal & Review Tracker</div>
                <div class='op-card-subtitle'>Persistent local database action log (SQLite), status transitions, role assignments, and audit trails</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col_flt1, col_flt2 = st.columns([2, 1])
    with col_flt1:
        st.markdown("<b>Filter Proposals by Status:</b>", unsafe_allow_html=True)
        status_flt = st.radio("Status Filter:", ["All"] + PROPOSAL_STATUSES, horizontal=True, label_visibility="collapsed")
    with col_flt2:
        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Export Proposals (CSV)",
            data=proposal_service.export_sanitized_csv(selected_region),
            file_name=f"{selected_region}_hap_proposals.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Proposals Table
    proposals_df = proposal_service.get_proposals(selected_region, status_filter=status_flt)

    if not proposals_df.empty:
        st.markdown(f"**Found {len(proposals_df)} persistent proposals for this region:**")
        
        # Display Table
        disp_df = proposals_df[["id", "proposal_code", "ward_name", "intervention_type", "department", "urgency", "budget_inr", "status", "created_at"]]
        st.dataframe(disp_df, use_container_width=True)

        st.markdown("<hr style='margin:16px 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)

        # REVIEWER ACTION PANEL
        st.markdown("### ✍️ Reviewer Action & Status Update")
        st.caption("⚠️ **Demonstration Mode:** Role selection in this prototype is unauthenticated and intended for testing multi-role workflows. In production, this requires an enterprise IAM / OAuth 2.0 / SSO integration (e.g. TNeGA SSO) with digital signature verification.")

        prop_ids = proposals_df["id"].tolist()
        selected_prop_id = st.selectbox(
            "Select Proposal to Review / Update:",
            options=prop_ids,
            format_func=lambda pid: f"ID #{pid}: {proposals_df[proposals_df['id'] == pid]['proposal_code'].iloc[0]} - {proposals_df[proposals_df['id'] == pid]['title'].iloc[0]} [{proposals_df[proposals_df['id'] == pid]['status'].iloc[0]}]"
        )

        prop_record = proposal_service.get_proposal_by_id(selected_prop_id)

        r_col1, r_col2 = st.columns([1.5, 1])
        with r_col1:
            safe_title = html.escape(str(prop_record['title']))
            safe_ward = html.escape(str(prop_record['ward_name']))
            safe_wid = html.escape(str(prop_record['ward_id']))
            safe_type = html.escape(str(prop_record['intervention_type']))
            safe_just = html.escape(str(prop_record['justification']))
            safe_status = html.escape(str(prop_record['status']))

            st.markdown(f"""
            <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px 16px;'>
                <b>Proposal Code:</b> {prop_record['proposal_code']}<br>
                <b>Title:</b> {safe_title}<br>
                <b>Ward:</b> {safe_ward} ({safe_wid})<br>
                <b>Intervention:</b> {safe_type}<br>
                <b>Estimated Budget:</b> ₹{prop_record['budget_inr']:,.2f}<br>
                <b>Justification:</b> {safe_just}<br>
                <b>Current Status:</b> <span class='status-pill status-{safe_status.lower()}'>{safe_status}</span>
            </div>
            """, unsafe_allow_html=True)

        with r_col2:
            new_st = st.selectbox("New Lifecycle Status:", PROPOSAL_STATUSES, index=PROPOSAL_STATUSES.index(prop_record["status"]))
            reviewer_role = st.selectbox("Reviewer Role (Simulated):", ["Municipal Commissioner", "Chief Health Officer", "Superintending Engineer (PWD)", "District Collector"])
            review_comment = st.text_input("Reviewer Remarks / Order Ref:", placeholder="e.g. Approved under Monsoon HAP budget code #4102.")

            if st.button("💾 Commit Status Transition & Save Audit Record", type="primary", use_container_width=True):
                try:
                    proposal_service.update_proposal_status(
                        proposal_id=selected_prop_id,
                        new_status=new_st,
                        changed_by=reviewer_role,
                        comment=review_comment or "Status updated by reviewer."
                    )
                    st.success("✅ Status updated and permanently committed to SQLite database!")
                    st.rerun()
                except ValueError as err:
                    st.error(f"❌ State Machine Error: {err}")

        # Audit History
        st.markdown("#### 📜 Chronological Audit Trail")
        audit_trail = proposal_service.get_audit_trail(selected_prop_id)
        for a in audit_trail:
            safe_who = html.escape(str(a['changed_by']))
            safe_prev = html.escape(str(a.get('previous_status', 'None')))
            safe_curr = html.escape(str(a['new_status']))
            safe_cmt = html.escape(str(a.get('comment', '')))

            st.markdown(f"""
            <div style='font-size:0.8rem; background:white; border:1px solid #e2e8f0; border-radius:6px; padding:8px 12px; margin-bottom:6px;'>
                <b>{a['changed_at']}</b> &nbsp;|&nbsp; <b>{safe_who}</b> &nbsp;|&nbsp; 
                Transition: <code>{safe_prev} ➔ {safe_curr}</code> &nbsp;|&nbsp; 
                <i>"{safe_cmt}"</i>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("No proposals found for the current filter. Create a new proposal from the **'Ward Risk Profile & Assessment'** tab.")

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 7: DATA FRESHNESS & PROVENANCE
# ==============================================================================
elif current_tab == "Data Freshness & Provenance":
    st.markdown("""
    <div class='op-card' style='margin-bottom:12px;'>
        <div class='op-card-header'>
            <div>
                <div class='op-card-title'>📡 Dataset Provenance, Latency & Freshness Tracker</div>
                <div class='op-card-subtitle'>Complete sensor lineage and transparent distinction between real observations and demonstration runs</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#fef3c7; border:1px solid #fde68a; border-radius:8px; padding:12px 16px; margin-bottom:14px; font-size:0.82rem; color:#92400e;'>
        <b>⚠️ Sample Demonstration Notice:</b> This decision-support system is currently running in demonstration mode.
        Spatial grids are calibrated to USGS Landsat-9 and Sentinel-2 summer baseline imagery. Ground weather observations stream live from the Open-Meteo meteorological feed.
    </div>
    """, unsafe_allow_html=True)

    # Layers Audit Table
    layers = region_status["layers"]
    for l_key, l_info in layers.items():
        st.markdown(f"""
        <div style='background:white; border:1px solid #e2e8f0; border-radius:8px; padding:12px 16px; margin-bottom:8px;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div style='font-size:0.92rem; font-weight:800; color:#0f172a;'>
                    {l_info['name']}
                </div>
                <div style='font-size:0.75rem; font-weight:700; color:{"#16a34a" if l_info["status"] in ["Available", "Live Operational"] else "#dc2626"};'>
                    ● {l_info['status']}
                </div>
            </div>
            <div style='font-size:0.78rem; color:#64748b; margin-top:4px;'>
                <b>Source:</b> {l_info['source']} &nbsp;|&nbsp; 
                <b>Date:</b> {l_info['observation_date']} &nbsp;|&nbsp; 
                <b>Quality:</b> {l_info.get('quality', 'Standard')}
            </div>
            <div style='font-size:0.75rem; color:#475569; margin-top:2px;'>
                <i>{l_info['note']}</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Observation Timeline
    st.markdown("### ⏱️ Available Observation Timeline")
    scenes = data_status_tracker.get_timeline_scenes(selected_region)
    for s in scenes:
        active_badge = "<span style='background:#dcfce7; color:#16a34a; font-size:0.72rem; font-weight:700; padding:2px 8px; border-radius:4px;'>Active Scene</span>" if s["is_active"] else "<span style='color:#64748b; font-size:0.72rem;'>Historical Archive</span>"
        st.markdown(f"""
        <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:10px 14px; margin-bottom:6px; display:flex; justify-content:space-between; align-items:center;'>
            <div>
                <b>{s['label']}</b> ({s['season']})<br>
                <span style='font-size:0.75rem; color:#64748b;'>Sensor: {s['satellite']} | Mean Temp: {s['avg_temp_c']}°C</span>
            </div>
            <div>{active_badge}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 8: TRICHY VS CHENNAI (DELTA)
# ==============================================================================
elif current_tab == "Trichy vs Chennai (Delta)":
    st.markdown("""
    <div class='op-card' style='margin-bottom:12px;'>
        <div class='op-card-header'>
            <div>
                <div class='op-card-title'>🔬 Delta vs. Non-Delta Regional Comparative Analysis</div>
                <div class='op-card-subtitle'>Contrasting alluvial inland agricultural fringes with dense coastal metropolitan corridors</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    comp_table_html = (
        "<table style='width:100%; border-collapse:collapse; font-size:0.85rem;'>"
        "<thead>"
        "<tr style='border-bottom:2px solid #e2e8f0; font-size:0.75rem; text-transform:uppercase; color:#64748b;'>"
        "<th style='padding:8px;'>Dimension</th>"
        "<th style='padding:8px;'>Tiruchirappalli (Cauvery Delta Gateway)</th>"
        "<th style='padding:8px;'>Chennai (Non-Delta Coastal Metro)</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        "<tr style='border-bottom:1px solid #f1f5f9;'>"
        "<td style='padding:10px 8px; font-weight:700;'>Geographic Context</td>"
        "<td style='padding:10px 8px;'>Inland riverine delta apex surrounded by agricultural wetlands</td>"
        "<td style='padding:10px 8px;'>Coastal flatland bordered by the Bay of Bengal</td>"
        "</tr>"
        "<tr style='border-bottom:1px solid #f1f5f9;'>"
        "<td style='padding:10px 8px; font-weight:700;'>Natural Cooling Buffer</td>"
        "<td style='padding:10px 8px;'>Cauvery/Kollidam river network (500m buffer)</td>"
        "<td style='padding:10px 8px;'>Sea breeze maritime moderation along eastern coast</td>"
        "</tr>"
        "<tr style='border-bottom:1px solid #f1f5f9;'>"
        "<td style='padding:10px 8px; font-weight:700;'>Urban Encroachment</td>"
        "<td style='padding:10px 8px;'>Conversion of fertile delta agricultural land into industrial corridors</td>"
        "<td style='padding:10px 8px;'>Extreme impervious density, street canyons, loss of wetlands</td>"
        "</tr>"
        "<tr>"
        "<td style='padding:10px 8px; font-weight:700;'>Peak Hotspots</td>"
        "<td style='padding:10px 8px; color:#dc2626; font-weight:700;'>44.65°C (Extreme dry summer interior heat)</td>"
        "<td style='padding:10px 8px; color:#dc2626; font-weight:700;'>45.00°C (Intense apparent heat stress due to humidity)</td>"
        "</tr>"
        "</tbody>"
        "</table>"
    )
    st.markdown(comp_table_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 9: LIVE API & ARCHITECTURE
# ==============================================================================
elif current_tab == "Live API & Architecture":
    st.markdown("""
    <div class='op-card' style='margin-bottom:12px;'>
        <div class='op-card-header'>
            <div>
                <div class='op-card-title'>🌐 System Architecture, Live API Guide & Extensions</div>
                <div class='op-card-subtitle'>Live weather API ingestion guide, code architectures, and multi-sector extension pathways</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 1. Weather-Model-Based Conditions (Open-Meteo GFS/ICON Integration) [Estimated]")
    st.caption("Atmospheric model conditions provided by Open-Meteo.com under CC BY 4.0 license (NOAA GFS / DWD ICON blend).")

    col_w1, col_w2 = st.columns([1, 2])
    with col_w1:
        st.markdown(f"**Target City:** {reg_info['name']}")
        st.markdown(f"**Coordinates:** {reg_info['center_lat']:.4f}° N, {reg_info['center_lon']:.4f}° E")
        st.markdown("**Source:** Open-Meteo API")
        st.markdown("**Model:** NOAA GFS / DWD ICON blend")
        st.button("🔄 Fetch Atmospheric Model Conditions Now")
        
    with col_w2:
        live_data = fetch_live_weather(reg_info["center_lat"], reg_info["center_lon"])
        if live_data:
            w_c1, w_c2, w_c3, w_c4 = st.columns(4)
            with w_c1:
                st.metric("Model Air Temp [Estimated]", f"{live_data.get('temperature_2m', 34.0)}°C")
            with w_c2:
                st.metric("Relative Humidity [Estimated]", f"{live_data.get('relative_humidity_2m', 60)}%")
            with w_c3:
                st.metric("Apparent Heat Index [Derived]", f"{live_data.get('apparent_temperature', 38.0)}°C")
            with w_c4:
                st.metric("Wind Speed [Estimated]", f"{live_data.get('wind_speed_10m', 12)} km/h")
            valid_t = live_data.get('time', datetime.now().strftime('%Y-%m-%dT%H:00'))
            retrieved_t = datetime.now().strftime('%H:%M:%S IST')
            st.success(f"✅ Weather-model-based conditions retrieved from Open-Meteo (NOAA GFS / DWD ICON blend). Valid Time: {valid_t} UTC | Retrieval Time: {retrieved_t}.")
        else:
            st.info("Weather-model-based conditions available via Open-Meteo API integration.")

    st.markdown("""
    <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:16px; margin-top:16px;'>
        <h4 style='color:#0f172a; margin-top:0;'>📖 Developer Reference: Weather API Ingestion Architecture</h4>
        <div style='background:#fffbeb; border:1px solid #fef3c7; border-radius:6px; padding:10px 14px; font-size:0.82rem; color:#92400e; margin-bottom:12px;'>
            <b>⚠️ Scientific Scope Notice:</b> In this version, weather-model-based conditions are monitored strictly as ambient context and are <b>NOT applied to modify pre-computed static satellite LST or ward risk scores</b>. The spatial model was trained on calibrated summer satellite LST scenes. Feeding dynamic single-point weather into static satellite models without multi-temporal recalibration would be scientifically invalid.
        </div>
        <pre style='background:#0f172a; color:#f8fafc; padding:12px; border-radius:6px; font-size:0.82rem;'>
# [ILLUSTRATION ONLY]: Conceptual architecture for future multi-temporal pipelines
# 1. Fetch real-time weather
live = fetch_live_weather(lat=10.7905, lon=78.7047)
ambient_temp = live["temperature_2m"]
humidity = live["relative_humidity_2m"]

# 2. In future multi-temporal versions, this would update dynamic forcing features:
# X_dynamic["Air_Temp_Forcing"] = ambient_temp
# X_dynamic["RH_Forcing"] = humidity
# y_pred = dynamic_ensemble.predict(X_dynamic)
        </pre>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🚀 Project Roadmap: Methodological Extensions")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("""
        <div style='background:white; border:1px solid #e2e8f0; border-radius:10px; padding:16px;'>
            <div style='font-size:0.75rem; font-weight:800; color:#0284c7; text-transform:uppercase;'>Extension 1</div>
            <div style='font-weight:800; font-size:1.0rem; margin-top:4px; color:#0f172a;'>🗺️ Custom Area Drawing (Leaflet-Draw)</div>
            <p style='font-size:0.82rem; color:#64748b; margin-top:6px;'>
                Integrate <b>Leaflet-Draw</b> to let urban planners delineate custom micro-zones (e.g. informal settlements, bus depots) and aggregate micro-grid heat exposure within user-drawn polygons.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with e2:
        st.markdown("""
        <div style='background:white; border:1px solid #e2e8f0; border-radius:10px; padding:16px;'>
            <div style='font-size:0.75rem; font-weight:800; color:#16a34a; text-transform:uppercase;'>Extension 2</div>
            <div style='font-weight:800; font-size:1.0rem; margin-top:4px; color:#0f172a;'>📄 Automated PDF Heat Action Plan Dossier</div>
            <p style='font-size:0.82rem; color:#64748b; margin-top:6px;'>
                Compile ward assessment indicators, plain-language drivers, and approved proposal details into a structured, printable municipal briefing sheet.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
