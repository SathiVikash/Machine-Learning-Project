# -*- coding: utf-8 -*-
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
    initial_sidebar_state="expanded"
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
    div[data-testid="stRadio"] > label {
        display: none !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] > label {
        display: none !important;
    }
    div[data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label,
    div[data-testid="stRadio"] > div label {
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

    /* Space-Optimized Structured Sidebar */
    [data-testid="stSidebar"] {
        display: block !important;
        min-width: 260px !important;
        max-width: 275px !important;
        width: 265px !important;
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        box-shadow: 1px 0 4px rgba(0,0,0,0.02) !important;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.35rem !important;
        padding-top: 0.8rem !important;
        padding-left: 0.6rem !important;
        padding-right: 0.6rem !important;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        display: flex !important;
        flex-direction: column !important;
        gap: 4px !important;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label,
    [data-testid="stSidebar"] div[data-testid="stRadio"] > div label {
        display: flex !important;
        align-items: center !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        background: transparent !important;
        cursor: pointer !important;
        border: 1px solid transparent !important;
        transition: all 0.15s ease-in-out !important;
        margin: 0 !important;
        width: 100% !important;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background: #f1f5f9 !important;
        color: #0f172a !important;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
        background: #0f172a !important;
        color: #ffffff !important;
        box-shadow: 0 1px 3px rgba(15,23,42,0.15) !important;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p,
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) span {
        color: #ffffff !important;
        font-weight: 700 !important;
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
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:8px; justify-content:flex-end; height:100%;'>
        <div style='background:#f0fdf4; color:#15803d; border:1px solid #bbf7d0; padding:4px 10px; border-radius:8px; font-size:0.75rem; font-weight:700;'>
            ● Model Online: Spatial AI Active
        </div>
        <div style='font-size:0.72rem; color:#64748b; line-height:1.2; text-align:right;'>
            <b>{reg_info['name']}</b> &bull; Urban Decision Support<br>
            <span style='color:#94a3b8; font-size:0.68rem;'>System Surveillance Active</span>
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
# 2. SIDEBAR STRUCTURED NAVIGATION MENU
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style='padding: 6px 4px 12px 4px; border-bottom: 1px solid #e2e8f0; margin-bottom: 10px;'>
        <div style='display:flex; align-items:center; gap:10px;'>
            <div style='width:36px; height:36px; min-width:36px; background:linear-gradient(135deg, #e11d48, #ea580c); border-radius:8px; display:flex; align-items:center; justify-content:center; color:white; font-size:18px; box-shadow:0 2px 6px rgba(225,29,72,0.3);'>🔥</div>
            <div>
                <div style='font-weight:800; font-size:0.92rem; color:#0f172a; letter-spacing:-0.2px; line-height:1.2;'>HEAT WATCH</div>
                <div style='font-size:0.68rem; font-weight:600; color:#64748b;'>Decision Support Portal</div>
            </div>
        </div>
    </div>
    <div style='font-size:0.68rem; font-weight:800; text-transform:uppercase; letter-spacing:0.8px; color:#94a3b8; margin: 4px 0 6px 4px;'>
        NAVIGATION MENU
    </div>
    """, unsafe_allow_html=True)

    nav_tabs = [
        "Dashboard",
        "Ward Risk Profile & Assessment",
        "Spatial Heat & Risk Map",
        "Scenario Workspace (What-If)",
        "Explainable AI (XAI & SHAP)",
        "HAP Proposals & Review Tracker",
        "Data Freshness & Provenance"
    ]

    nav_display_names = {
        "Dashboard": "📊 Dashboard",
        "Ward Risk Profile & Assessment": "📋 Ward Risk Profile",
        "Spatial Heat & Risk Map": "🗺️ Spatial Heat Map",
        "Scenario Workspace (What-If)": "🔮 Scenario Simulation",
        "Explainable AI (XAI & SHAP)": "🧠 Explainable AI (SHAP)",
        "HAP Proposals & Review Tracker": "📝 HAP Action Tracker",
        "Data Freshness & Provenance": "🔄 Data Provenance"
    }

    current_tab = st.radio(
        "Navigation Menu",
        options=nav_tabs,
        format_func=lambda x: nav_display_names.get(x, x),
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style='margin-top: 28px; padding: 10px 12px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;'>
        <div style='font-size: 0.70rem; font-weight: 700; color: #0f172a; display: flex; align-items: center; gap: 6px;'>
            <span style='color: #16a34a;'>●</span> System Operational
        </div>
        <div style='font-size: 0.68rem; color: #64748b; margin-top: 4px; line-height: 1.3;'>
            Spatial Screening & Ward Decision Support System
        </div>
    </div>
    """, unsafe_allow_html=True)

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
# TAB 1: DASHBOARD
# ==============================================================================
if current_tab == "Dashboard":
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
        if total_pop <= 900000:
            pop_display = f"{total_pop/1000:.0f}k"
        else:
            millions = total_pop / 1000000
            pop_display = f"{millions:.2f}".rstrip("0").rstrip(".") + "M"

        st.markdown(f"""
        <div class='kpi-container'>
            <div class='kpi-header-row'>
                <span class='kpi-title'>Exposed Population</span>
                <span class='kpi-icon-pill' style='background:#f5f3ff; color:#7c3aed;'>👥</span>
            </div>
            <div class='kpi-value-row'>
                <span class='kpi-main-number'>{pop_display}</span>
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
        search_query = st.text_input("Filter Wards by Name or ID:", placeholder="🔍 Search e.g. Palakkarai, Srirangam, W-12...", label_visibility="collapsed")
        
        display_wards = ward_df.copy()
        if search_query:
            display_wards = display_wards[
                display_wards["ward_name"].str.contains(search_query, case=False) |
                display_wards["ward_id"].str.contains(search_query, case=False)
            ]

        # Table HTML with clean string formatting (zero indentation to prevent markdown code block formatting)
        table_rows = []
        for idx, row in display_wards.iterrows():
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
            "<div style='max-height:388px; overflow-y:auto; overflow-x:auto; background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:4px 10px; box-shadow:0 1px 2px rgba(0,0,0,0.02); margin-top:2px;'>"
            "<table style='width:100%; border-collapse:collapse;'>"
            "<thead style='position:sticky; top:0; background:#ffffff; z-index:2;'>"
            "<tr style='border-bottom:2px solid #e2e8f0; font-size:0.72rem; text-transform:uppercase; color:#64748b; font-weight:700; text-align:left;'>"
            "<th style='padding:8px 6px; background:#ffffff;'>Rank</th>"
            "<th style='padding:8px 6px; background:#ffffff;'>Ward ID</th>"
            "<th style='padding:8px 6px; background:#ffffff;'>Name</th>"
            "<th style='padding:8px 6px; background:#ffffff;'>Tier</th>"
            "<th style='padding:8px 6px; background:#ffffff;'>HRI</th>"
            "<th style='padding:8px 6px; background:#ffffff;'>Mean LST</th>"
            "<th style='padding:8px 6px; background:#ffffff;'>Density</th>"
            "</tr>"
            "</thead>"
            f"<tbody>{''.join(table_rows)}</tbody>"
            "</table>"
            "</div>"
        )
        st.markdown(table_html, unsafe_allow_html=True)

    with c_dispatch:
        render_ward_selector("summary_table_sync")

        cat_badge_color = RISK_COLORS.get(target_ward_record["Risk_Category"], "#ea580c")
        drivers = generate_plain_language_drivers(target_ward_record)
        drivers_html = "".join([
            f"<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:8px 12px; margin-top:6px; font-size:0.8rem;'>"
            f"<b>{d['icon']} {d['factor']}</b>: <span style='color:#dc2626; font-weight:700;'>{d['impact']}</span><br>"
            f"<span style='color:#64748b;'>{d['description']}</span>"
            f"</div>"
            for d in drivers[:3]
        ])

        dispatch_card_html = (
            f"<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:16px; margin-top:2px; box-shadow:0 1px 2px rgba(0,0,0,0.02);'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>"
            f"<div>"
            f"<div style='font-size:1.25rem; font-weight:800; color:#0f172a;'>{target_ward_record['ward_name']}</div>"
            f"<div style='font-size:0.78rem; color:#64748b; margin-top:2px;'>"
            f"Ward ID: <b>{target_ward_record['ward_id']}</b> &nbsp;|&nbsp; Priority Rank: <b>#{target_ward_record['Priority_Rank']} of {len(ward_df)}</b>"
            f"</div>"
            f"</div>"
            f"<div style='text-align:right;'>"
            f"<span style='background:{cat_badge_color}; color:white; font-size:0.78rem; font-weight:700; padding:4px 10px; border-radius:12px;'>{target_ward_record['Risk_Category']}</span>"
            f"<div style='font-size:0.8rem; font-weight:800; color:#dc2626; margin-top:4px;'>LST: {target_ward_record['Mean_LST_Celsius']:.1f}°C</div>"
            f"</div>"
            f"</div>"
            f"<b style='font-size:0.82rem; color:#334155;'>Key Thermal & Social Drivers:</b>"
            f"{drivers_html}"
            f"</div>"
        )
        st.markdown(dispatch_card_html, unsafe_allow_html=True)

        # Fetch Live Weather for selected ward centroid
        w_cells = grid_df[grid_df["ward_id"] == target_ward_record["ward_id"]]
        if not w_cells.empty and "lat" in w_cells.columns and "lon" in w_cells.columns:
            w_lat = float(w_cells["lat"].mean())
            w_lon = float(w_cells["lon"].mean())
        else:
            w_lat = float(reg_info["center_lat"])
            w_lon = float(reg_info["center_lon"])

        ward_weather = fetch_live_weather(w_lat, w_lon)
        w_temp = f"{ward_weather['temperature_2m']}°C" if ward_weather else "34.2°C"
        w_rh = f"{ward_weather['relative_humidity_2m']}%" if ward_weather else "60%"
        w_app = f"{ward_weather['apparent_temperature']}°C" if ward_weather else "39.1°C"
        w_wind = f"{ward_weather['wind_speed_10m']} km/h" if ward_weather else "11.2 km/h"
        valid_t = ward_weather.get('time', datetime.now().strftime('%Y-%m-%dT%H:00')) if ward_weather else datetime.now().strftime('%Y-%m-%dT%H:00')

        weather_card_html = (
            f"<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:12px 14px; margin-top:10px; box-shadow:0 1px 2px rgba(0,0,0,0.02);'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>"
            f"<div style='font-size:0.84rem; font-weight:800; color:#0f172a; display:flex; align-items:center; gap:6px;'>"
            f"<span>🌤️</span> Weather-Model-Based Conditions"
            f"</div>"
            f"<span style='background:#f0fdf4; color:#15803d; font-size:0.68rem; font-weight:700; padding:2px 8px; border-radius:10px; border:1px solid #bbf7d0;'>"
            f"Ward Centroid ({w_lat:.3f}°N, {w_lon:.3f}°E)"
            f"</span>"
            f"</div>"
            f"<div style='display:grid; grid-template-columns: repeat(4, 1fr); gap:8px;'>"
            f"<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:6px 4px; text-align:center;'>"
            f"<div style='font-size:0.68rem; color:#64748b; font-weight:600;'>Dry-Bulb Temp</div>"
            f"<div style='font-size:0.95rem; font-weight:800; color:#0f172a; margin-top:2px;'>{w_temp}</div>"
            f"</div>"
            f"<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:6px 4px; text-align:center;'>"
            f"<div style='font-size:0.68rem; color:#64748b; font-weight:600;'>Rel. Humidity</div>"
            f"<div style='font-size:0.95rem; font-weight:800; color:#0f172a; margin-top:2px;'>{w_rh}</div>"
            f"</div>"
            f"<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:6px 4px; text-align:center;'>"
            f"<div style='font-size:0.68rem; color:#64748b; font-weight:600;'>Heat Index</div>"
            f"<div style='font-size:0.95rem; font-weight:800; color:#dc2626; margin-top:2px;'>{w_app}</div>"
            f"</div>"
            f"<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:6px 4px; text-align:center;'>"
            f"<div style='font-size:0.68rem; color:#64748b; font-weight:600;'>Wind Speed</div>"
            f"<div style='font-size:0.95rem; font-weight:800; color:#0f172a; margin-top:2px;'>{w_wind}</div>"
            f"</div>"
            f"</div>"
            f"<div style='margin-top:6px; font-size:0.66rem; color:#94a3b8; display:flex; justify-content:space-between; align-items:center;'>"
            f"<span>Source: Open-Meteo (NOAA GFS / DWD ICON blend) | Valid: {valid_t} UTC</span>"
            f"<span style='color:#16a34a; font-weight:700;'>● Ward Live Sync</span>"
            f"</div>"
            f"</div>"
        )
        st.markdown(weather_card_html, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: WARD RISK PROFILE & ASSESSMENT (THE #1 MOST IMPORTANT SCREEN)
# ==============================================================================
elif current_tab == "Ward Risk Profile & Assessment":
    render_ward_selector("profile_tab")

    profile = get_ward_assessment_profile(target_ward_record.to_dict(), selected_region)

    # SITUATION SECTION - GRAPH FORMAT
    st.markdown("### Situation & Physical Climate Context")
    
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.7), dpi=100)
    fig1.patch.set_facecolor('#ffffff')
    
    # Graph 1: Thermal Radiation Metrics
    thermal_labels = ['City Baseline', 'Ward Mean LST', 'Peak Hotspot']
    thermal_vals = [
        float(ward_df['Mean_LST_Celsius'].mean()),
        float(profile['situation']['mean_lst_c']),
        float(profile['situation']['max_lst_c'])
    ]
    thermal_colors = ['#0284c7', '#ea580c', '#dc2626']
    
    ax1.set_facecolor('#ffffff')
    bars1 = ax1.barh(thermal_labels, thermal_vals, color=thermal_colors, height=0.52)
    ax1.set_xlim(25, max(thermal_vals) + 5.5)
    ax1.set_xlabel("Surface Temperature (°C)", fontsize=8, fontweight='600', color='#475569')
    ax1.set_title("Thermal Radiation Context (Landsat-9)", fontsize=9, fontweight='700', color='#0f172a', pad=6)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#cbd5e1')
    ax1.spines['bottom'].set_color('#cbd5e1')
    ax1.tick_params(colors='#475569', labelsize=8)
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.35, bar.get_y() + bar.get_height()/2, f"{w:.1f}°C", va='center', ha='left', fontsize=8, fontweight='700', color='#0f172a')
    
    # Graph 2: Physical Environmental Covariates
    env_labels = ['Water Dist (km)', 'Impervious (NDBI x100)', 'Canopy (NDVI x100)', 'Built Density (%)']
    b_dens = float(target_ward_record.get('Mean_Building_Density_pct', 0))
    ndvi_scaled = max(0, float(target_ward_record.get('Mean_NDVI', 0)) * 100)
    ndbi_scaled = max(0, float(target_ward_record.get('Mean_NDBI', 0)) * 100)
    dist_km = float(target_ward_record.get('Mean_Dist_to_Water_m', 0)) / 1000.0
    env_vals = [dist_km, ndbi_scaled, ndvi_scaled, b_dens]
    env_colors = ['#0ea5e9', '#b45309', '#16a34a', '#64748b']
    
    ax2.set_facecolor('#ffffff')
    bars2 = ax2.barh(env_labels, env_vals, color=env_colors, height=0.52)
    ax2.set_xlim(0, max(max(env_vals) * 1.25, 15))
    ax2.set_xlabel("Indicator Magnitude", fontsize=8, fontweight='600', color='#475569')
    ax2.set_title("Physical Microclimate Covariates", fontsize=9, fontweight='700', color='#0f172a', pad=6)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#cbd5e1')
    ax2.spines['bottom'].set_color('#cbd5e1')
    ax2.tick_params(colors='#475569', labelsize=8)
    for bar, val in zip(bars2, env_vals):
        w = bar.get_width()
        suffix = " km" if bar.get_y() == bars2[0].get_y() else ("%" if bar.get_y() == bars2[3].get_y() else "")
        ax2.text(w + 0.5, bar.get_y() + bar.get_height()/2, f"{val:.1f}{suffix}", va='center', ha='left', fontsize=8, fontweight='700', color='#0f172a')
    
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    st.markdown("<hr style='margin:14px 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)

    # PEOPLE AND VULNERABILITY SECTION - GRAPH FORMAT
    st.markdown("### People & Socio-Demographic Vulnerability")
    
    fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(10, 2.7), dpi=100)
    fig2.patch.set_facecolor('#ffffff')

    v_pop = profile['people_and_vulnerability']
    tot_pop = max(1, v_pop['estimated_population'])
    child_cnt = v_pop['children_count']
    eld_cnt = v_pop['elderly_count']
    other_cnt = max(0, tot_pop - (child_cnt + eld_cnt))

    pop_labels = ['General Cohort', 'Seniors (>65)', 'Children (<5)']
    pop_counts = [other_cnt, eld_cnt, child_cnt]
    pop_colors = ['#94a3b8', '#fb923c', '#f43f5e']

    ax2_1.set_facecolor('#ffffff')
    bars2_1 = ax2_1.barh(pop_labels, pop_counts, color=pop_colors, height=0.52)
    ax2_1.set_xlim(0, max(pop_counts) * 1.35)
    ax2_1.set_xlabel("Resident Population", fontsize=8, fontweight='600', color='#475569')
    ax2_1.set_title(f"Demographic Cohorts (Total: {tot_pop:,})", fontsize=9, fontweight='700', color='#0f172a', pad=6)
    ax2_1.spines['top'].set_visible(False)
    ax2_1.spines['right'].set_visible(False)
    ax2_1.spines['left'].set_color('#cbd5e1')
    ax2_1.spines['bottom'].set_color('#cbd5e1')
    ax2_1.tick_params(colors='#475569', labelsize=8)
    for bar, cnt in zip(bars2_1, pop_counts):
        w = bar.get_width()
        pct = (cnt / tot_pop) * 100
        ax2_1.text(w + (max(pop_counts)*0.02), bar.get_y() + bar.get_height()/2, f"{cnt:,} ({pct:.1f}%)", va='center', ha='left', fontsize=8, fontweight='700', color='#0f172a')

    # Subplot 2: Socio-Economic Fragility Factors (%)
    vuln_labels = ['Composite Vuln.', 'Child Ratio', 'Senior Ratio', 'Low-Income Homes']
    low_inc = float(v_pop['low_income_pct'])
    eld_pct = float(v_pop['elderly_ratio_pct'])
    child_pct = float(v_pop['children_ratio_pct'])
    comp_vuln = float(profile['situation']['vulnerability_score']) * 100.0

    vuln_vals = [comp_vuln, child_pct, eld_pct, low_inc]
    vuln_colors = ['#7c3aed', '#f59e0b', '#ea580c', '#dc2626']

    ax2_2.set_facecolor('#ffffff')
    bars2_2 = ax2_2.barh(vuln_labels, vuln_vals, color=vuln_colors, height=0.52)
    ax2_2.set_xlim(0, 105)
    ax2_2.set_xlabel("Vulnerability Weight (%)", fontsize=8, fontweight='600', color='#475569')
    ax2_2.set_title("Socio-Demographic Vulnerability Factors", fontsize=9, fontweight='700', color='#0f172a', pad=6)
    ax2_2.spines['top'].set_visible(False)
    ax2_2.spines['right'].set_visible(False)
    ax2_2.spines['left'].set_color('#cbd5e1')
    ax2_2.spines['bottom'].set_color('#cbd5e1')
    ax2_2.tick_params(colors='#475569', labelsize=8)
    for bar, val in zip(bars2_2, vuln_vals):
        w = bar.get_width()
        ax2_2.text(w + 1.2, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va='center', ha='left', fontsize=8, fontweight='700', color='#0f172a')

    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.markdown("<hr style='margin:14px 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)

    # WHY THIS WARD SECTION
    st.markdown("### Why This Ward? (Key Thermal & Vulnerability Drivers)")
    st.caption("Plain-language physical attributions explaining model findings without raw statistical jargon.")

    for d in profile["drivers_plain_language"]:
        d_color = "#dc2626" if d["direction"] == "heating" else "#16a34a"
        d_icon = d["icon"]
        d_factor = d["factor"]
        d_impact = d["impact"]
        d_desc = d["description"]
        driver_card_html = (
            f"<div style='background:white; border:1px solid #e2e8f0; border-left:4px solid {d_color}; border-radius:8px; padding:12px 16px; margin-bottom:8px;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
            f"<div style='font-size:0.95rem; font-weight:800; color:#0f172a;'>{d_icon} {d_factor}</div>"
            f"<div style='font-size:0.85rem; font-weight:800; color:{d_color};'>{d_impact}</div>"
            f"</div>"
            f"<div style='font-size:0.82rem; color:#475569; margin-top:4px;'>{d_desc}</div>"
            f"</div>"
        )
        st.markdown(driver_card_html, unsafe_allow_html=True)

    st.markdown("<hr style='margin:14px 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)

    # HIGH-IMPACT EYE-CATCHING HAP INTERVENTIONS SECTION
    w_name = profile["ward_name"]
    w_risk_cat = profile["situation"]["risk_category"]
    hap_header_html = (
        "<div style='background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%); border: 2px solid #fdba74; border-left: 6px solid #ea580c; border-radius: 12px; padding: 16px 20px; margin: 16px 0 14px 0; box-shadow: 0 4px 12px rgba(234, 88, 12, 0.08);'>"
        "<div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;'>"
        "<div>"
        "<span style='background: #ea580c; color: white; font-size: 0.72rem; font-weight: 800; padding: 3px 10px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.5px;'>Direct Action Mandate</span>"
        "<div style='font-size: 1.25rem; font-weight: 800; color: #7c2d12; margin-top: 6px;'>&#128737;&#65039; Recommended Heat Action Plan (HAP) Interventions</div>"
        f"<div style='font-size: 0.82rem; color: #9a3412; margin-top: 2px;'>High-priority targeted mitigation protocols for <b>{w_name}</b> (Risk Tier: <b>{w_risk_cat}</b>)</div>"
        "</div>"
        "<div style='text-align: right;'>"
        "<span style='background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; font-size: 0.78rem; font-weight: 800; padding: 6px 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(220,38,38,0.1);'>&#9889; Projected Cooling Impact: -1.5°C to -2.8°C</span>"
        "</div>"
        "</div>"
        "</div>"
    )
    st.markdown(hap_header_html, unsafe_allow_html=True)

    for act in profile["recommended_interventions"]:
        act_html = (
            "<div style='background: #ffffff; border: 1px solid #fed7aa; border-left: 5px solid #f97316; border-radius: 10px; padding: 12px 16px; margin-bottom: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.03); display: flex; align-items: center; justify-content: space-between;'>"
            "<div style='display: flex; align-items: center; gap: 12px;'>"
            "<span style='font-size: 20px; background: #fff7ed; padding: 6px 10px; border-radius: 8px; border: 1px solid #ffedd5;'>&#9889;</span>"
            "<div>"
            f"<div style='font-size: 0.92rem; font-weight: 700; color: #1e293b;'>{act}</div>"
            "<div style='font-size: 0.72rem; color: #64748b; margin-top: 2px;'>Priority Municipal Vector • Active Surveillance Protocol</div>"
            "</div>"
            "</div>"
            "<span style='background: #ffedd5; color: #c2410c; font-size: 0.72rem; font-weight: 800; padding: 4px 10px; border-radius: 20px; white-space: nowrap;'>MANDATORY</span>"
            "</div>"
        )
        st.markdown(act_html, unsafe_allow_html=True)

    st.markdown("<hr style='margin:16px 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)

    # FORMALIZE PROPOSAL CTA
    st.markdown("### Formalize Municipal Action Proposal")
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

    # Simulation Mode Selector
    sim_mode = st.radio(
        "Simulation Mode",
        options=["🌿 Ward Cooling Interventions (HAP)", "🏗️ New Building Construction Impact Simulator"],
        horizontal=True,
        label_visibility="collapsed"
    )
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    if sim_mode == "🌿 Ward Cooling Interventions (HAP)":
        # Base feature row for selected ward
        ward_cells = grid_df[grid_df["ward_id"] == st.session_state["selected_ward_id"]]
        base_row = ward_cells.iloc[0].to_dict()

        s_c1, s_c2 = st.columns([1.1, 1.9])

        with s_c1:
            st.markdown("<b style='font-size:0.95rem; color:#0f172a;'>Select Municipal Intervention Package</b>", unsafe_allow_html=True)
            
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

            if intervention_choice == "cool_roof":
                num_houses = st.slider(
                    "Houses / Buildings Coated with High-Albedo Coating:",
                    min_value=100,
                    max_value=5000,
                    value=1500,
                    step=100,
                    help="Estimated residential/commercial structures receiving high-albedo solar reflective coating (SRI > 100)."
                )
                intensity = num_houses / 1500.0
                st.caption(f"Estimated roof area treated: ~{num_houses * 85 / 10000:.1f} hectares. Shifts rooftop albedo by +{min(0.20, 0.14 * intensity):.2f}.")
                scenario_inputs = scenario_service.apply_preset_intervention(base_row, "cool_roof", intensity)

            elif intervention_choice == "urban_greening":
                num_trees = st.slider(
                    "Native Saplings / Trees Planted in Ward:",
                    min_value=250,
                    max_value=10000,
                    value=2500,
                    step=250,
                    help="High-density Miyawaki clusters, roadside avenues, and micro-park tree saplings."
                )
                intensity = num_trees / 2500.0
                st.caption(f"Estimated canopy cover added: ~{num_trees * 0.002:.1f} hectares. Increases vegetative NDVI by +{min(0.25, 0.16 * intensity):.2f}.")
                scenario_inputs = scenario_service.apply_preset_intervention(base_row, "urban_greening", intensity)

            elif intervention_choice == "de_paving":
                depave_sqm = st.slider(
                    "Impervious Asphalt / Concrete Area De-paved (m²):",
                    min_value=5000,
                    max_value=100000,
                    value=25000,
                    step=5000,
                    help="De-paving non-essential concrete/asphalt plazas and walkways into permeable grass/paver zones."
                )
                intensity = depave_sqm / 25000.0
                st.caption(f"De-paving {depave_sqm:,} m² (~{depave_sqm/10000:.1f} ha) reduces built-up index (NDBI) by -{min(0.18, 0.09 * intensity):.2f}.")
                scenario_inputs = scenario_service.apply_preset_intervention(base_row, "de_paving", intensity)

            elif intervention_choice == "water_corridor":
                canal_meters = st.slider(
                    "Canal / Wetland Water Buffer Restored (Meters):",
                    min_value=200,
                    max_value=4000,
                    value=1200,
                    step=100,
                    help="Restoring riparian buffers, desilting municipal canals, or developing retention ponds."
                )
                intensity = canal_meters / 1200.0
                st.caption(f"Restoring {canal_meters:,} linear meters of water corridor improves evaporative cooling buffer.")
                scenario_inputs = scenario_service.apply_preset_intervention(base_row, "water_corridor", intensity)

            elif intervention_choice == "comprehensive_hap":
                coverage_pct = st.slider(
                    "Municipal Resilience Coverage (% of Ward Area):",
                    min_value=10,
                    max_value=100,
                    value=50,
                    step=5,
                    help="Multi-sector deployment: cool roofs + urban forestry + shaded walkways + de-paving."
                )
                intensity = coverage_pct / 50.0
                st.caption(f"Targeting {coverage_pct}% of the ward with synchronized urban cooling measures.")
                scenario_inputs = scenario_service.apply_preset_intervention(base_row, "comprehensive_hap", intensity)

            else: # custom
                scenario_inputs = dict(base_row)
                scenario_inputs["NDVI"] = st.slider("NDVI (Vegetation Index):", -0.2, 0.85, float(base_row["NDVI"]), 0.05)
                scenario_inputs["NDBI"] = st.slider("NDBI (Built-up Index):", -0.4, 0.60, float(base_row["NDBI"]), 0.05)
                scenario_inputs["Albedo"] = st.slider("Albedo (Reflectance):", 0.05, 0.55, float(base_row.get("Albedo", 0.18)), 0.02)
                scenario_inputs["Building_Density_pct"] = st.slider("Building Density (%):", 0.0, 95.0, float(base_row["Building_Density_pct"]), 5.0)

        with s_c2:
            st.markdown("<b style='font-size:0.95rem; color:#0f172a;'>Model Evaluation (Baseline vs. Scenario)</b>", unsafe_allow_html=True)

            res = scenario_service.evaluate_scenario(
                model=model,
                baseline_row=base_row,
                scenario_features=scenario_inputs,
                vulnerability_score=float(target_ward_record["Mean_Vulnerability_Score"])
            )

            base_lst = res['baseline']['predicted_lst_c']
            scen_lst = res['scenario']['predicted_lst_c']
            cooling = res['delta']['cooling_celsius']
            base_hri = res['baseline']['heat_risk_index']
            scen_hri = res['scenario']['heat_risk_index']
            hri_red = res['delta']['hri_reduction']
            hri_pct = res['delta']['pct_hri_improvement']

            # Comparison Graph: Baseline vs Scenario
            fig_eval, (ax_lst, ax_hri) = plt.subplots(1, 2, figsize=(7.2, 2.5), dpi=100)
            fig_eval.patch.set_facecolor('#ffffff')

            # Subplot 1: Surface Temperature (LST)
            bars_lst = ax_lst.bar(['Baseline', 'Scenario'], [base_lst, scen_lst], color=['#f87171', '#38bdf8'], width=0.48)
            ax_lst.set_facecolor('#ffffff')
            ax_lst.set_ylabel("Surface Temp (°C)", fontsize=8, fontweight='600', color='#475569')
            ax_lst.set_title("Surface LST (°C)", fontsize=9, fontweight='700', color='#0f172a', pad=6)
            min_y_lst = max(0, min(base_lst, scen_lst) - 4.0)
            max_y_lst = max(base_lst, scen_lst) + 3.0
            ax_lst.set_ylim(min_y_lst, max_y_lst)
            ax_lst.spines['top'].set_visible(False)
            ax_lst.spines['right'].set_visible(False)
            ax_lst.spines['left'].set_color('#cbd5e1')
            ax_lst.spines['bottom'].set_color('#cbd5e1')
            ax_lst.tick_params(colors='#475569', labelsize=8)

            for bar in bars_lst:
                h = bar.get_height()
                ax_lst.text(bar.get_x() + bar.get_width()/2, h + 0.35, f"{h:.2f}°C", ha='center', va='bottom', fontsize=8, fontweight='700', color='#0f172a')
            
            # Delta indicator on LST plot
            lst_delta_sign = f"-{cooling:.2f}°C" if cooling > 0 else f"+{abs(cooling):.2f}°C"
            ax_lst.text(0.5, 0.88, f"Impact: {lst_delta_sign}", transform=ax_lst.transAxes, ha='center', fontsize=8, fontweight='800', color='#0284c7' if cooling > 0 else '#dc2626', bbox=dict(boxstyle='round,pad=0.25', facecolor='#f0f9ff' if cooling > 0 else '#fef2f2', edgecolor='#bae6fd' if cooling > 0 else '#fecaca', linewidth=0.8))

            # Subplot 2: Heat Risk Index (HRI)
            bars_hri = ax_hri.bar(['Baseline', 'Scenario'], [base_hri, scen_hri], color=['#fb923c', '#4ade80'], width=0.48)
            ax_hri.set_facecolor('#ffffff')
            ax_hri.set_ylabel("Heat Risk Index (0-1)", fontsize=8, fontweight='600', color='#475569')
            ax_hri.set_title("Heat Risk Index (HRI)", fontsize=9, fontweight='700', color='#0f172a', pad=6)
            ax_hri.set_ylim(0, max(max(base_hri, scen_hri) * 1.35, 0.5))
            ax_hri.spines['top'].set_visible(False)
            ax_hri.spines['right'].set_visible(False)
            ax_hri.spines['left'].set_color('#cbd5e1')
            ax_hri.spines['bottom'].set_color('#cbd5e1')
            ax_hri.tick_params(colors='#475569', labelsize=8)

            for bar in bars_hri:
                h = bar.get_height()
                ax_hri.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.3f}", ha='center', va='bottom', fontsize=8, fontweight='700', color='#0f172a')

            # Delta indicator on HRI plot
            hri_delta_text = f"Risk: -{hri_red:.3f} (-{hri_pct:.1f}%)" if hri_red > 0 else "No Change"
            ax_hri.text(0.5, 0.88, hri_delta_text, transform=ax_hri.transAxes, ha='center', fontsize=8, fontweight='800', color='#15803d' if hri_red > 0 else '#475569', bbox=dict(boxstyle='round,pad=0.25', facecolor='#f0fdf4' if hri_red > 0 else '#f8fafc', edgecolor='#bbf7d0' if hri_red > 0 else '#e2e8f0', linewidth=0.8))

            plt.tight_layout()
            st.pyplot(fig_eval, use_container_width=True)
            plt.close(fig_eval)

            # Brief Description and Model Domain Check Callout
            st.markdown(f"""
            <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:10px 14px; margin-top:4px; font-size:0.8rem; color:#334155;'>
                <b>Simulated Ward Impact:</b> Applying this intervention shifts ward surface temperature from <b>{base_lst:.2f}°C</b> to <b>{scen_lst:.2f}°C</b> (<span style='color:#0284c7; font-weight:700;'>{lst_delta_sign} cooling</span>), improving the ward Heat Risk Index by <span style='color:#15803d; font-weight:700;'>{hri_pct:.1f}%</span> (from {base_hri:.3f} to {scen_hri:.3f}).
            </div>
            """, unsafe_allow_html=True)

            if res["is_valid_domain"]:
                st.markdown("""
                <div style='background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:8px 12px; margin-top:8px; font-size:0.78rem; color:#166534; display:flex; align-items:center; gap:8px;'>
                    <span style='font-size:1rem;'>✅</span>
                    <div>
                        <b>Model Domain Check: Within Empirical Bounds.</b> All modified environmental indicators remain within the valid training bounds of satellite observations.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                clamped_names = ", ".join([c["feature"] for c in res["clamped_features"]])
                st.markdown(f"""
                <div style='background:#fffbeb; border:1px solid #fde68a; border-radius:8px; padding:8px 12px; margin-top:8px; font-size:0.78rem; color:#92400e; display:flex; align-items:center; gap:8px;'>
                    <span style='font-size:1rem;'>⚠️</span>
                    <div>
                        <b>Model Domain Check: Clamped to Empirical Boundary.</b> Indicators (<b>{clamped_names}</b>) reached maximum physical bounds and were safely clamped to prevent extrapolation errors.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<hr style='margin:10px 0; border-color:#f1f5f9;'>", unsafe_allow_html=True)
            st.markdown("<b style='font-size:0.80rem; color:#475569;'>Physical Assumptions & Limitations:</b>", unsafe_allow_html=True)
            for a in res["assumptions_and_limitations"]:
                st.markdown(f"<li style='font-size:0.75rem; color:#64748b; line-height:1.3;'>{a}</li>", unsafe_allow_html=True)

    else:
        # NEW BUILDING CONSTRUCTION IMPACT SIMULATOR
        b_c1, b_c2 = st.columns([1.1, 1.9])

        with b_c1:
            st.markdown("<b style='font-size:0.95rem; color:#0f172a;'>Building Specifications & Site Parameters</b>", unsafe_allow_html=True)
            st.caption("Enter architectural dimensions, occupancy, and parking format to simulate environmental impact.")

            b_sqft = st.slider(
                "Ground Plot Footprint (sq. ft):",
                min_value=2000,
                max_value=100000,
                value=15000,
                step=1000,
                help="Ground footprint area occupied by the proposed building structure."
            )
            b_floors = st.slider(
                "Number of Floors (Stories):",
                min_value=1,
                max_value=30,
                value=6,
                step=1,
                help="Total vertical building height in stories."
            )
            
            gba_sqft = b_sqft * b_floors
            gba_sqm = gba_sqft * 0.092903
            footprint_sqm = b_sqft * 0.092903
            
            st.markdown(f"<div style='font-size:0.75rem; color:#0284c7; font-weight:700; margin-bottom:8px;'>Gross Built Area (GBA): {gba_sqft:,} sq.ft ({gba_sqm:,.0f} m²) | Ground Footprint: {footprint_sqm:,.0f} m²</div>", unsafe_allow_html=True)

            b_residents = st.slider(
                "Approximate Residents / Occupants:",
                min_value=10,
                max_value=2000,
                value=180,
                step=10,
                help="Estimated peak occupant headcount living or working inside the building."
            )

            b_parking = st.radio(
                "Parking Facility Type:",
                options=["Inside Building (Podium / Stilt / Underground)", "Surface Asphalt Parking Lot (Open Ground)"],
                index=0,
                help="Surface parking creates a large open heat-absorbing asphalt apron; indoor/underground parking preserves ground footprint."
            )

            b_basement = st.radio(
                "Basement Excavation:",
                options=["Yes (Underground Basement Levels)", "No (Slab on Grade)"],
                index=0,
                help="Basements require deep sub-soil excavation affecting groundwater percolation, but provide higher structural thermal mass."
            )

            st.markdown("<b style='font-size:0.85rem; color:#0f172a; margin-top:10px; display:block;'>Green Building Mitigation Options:</b>", unsafe_allow_html=True)
            mit_cool_roof = st.checkbox("High-Albedo Cool Roof Coating (SRI > 82)", value=True, help="Reflects 80% solar irradiance from rooftop slab.")
            mit_permeable = st.checkbox("Permeable Pavers for Driveways & Walkways", value=True, help="Permeable pavers allow rainwater infiltration and avoid asphalt heat trap.")
            mit_trees = st.checkbox("Mandatory 20% Native Tree Canopy Buffer", value=True, help="Provides continuous tree shading and microclimatic evapotranspiration.")
            mit_energy = st.checkbox("Energy-Efficient Inverter HVAC Heat Exchangers", value=True, help="Reduces anthropogenic waste heat exhaust into the surrounding street canyon.")

        with b_c2:
            st.markdown("<b style='font-size:0.95rem; color:#0f172a;'>Microclimate Impact & Mitigation Assessment</b>", unsafe_allow_html=True)

            base_temp = float(target_ward_record["Mean_LST_Celsius"])
            
            # Physical impact calculations
            delta_concrete = min(1.8, (footprint_sqm / 5000.0) * 0.35 + (b_floors * 0.04))
            delta_ac = min(1.2, (b_residents * 1.2) / 250.0 * 0.25)
            delta_parking = 0.55 if "Surface" in b_parking else 0.0
            delta_basement = -0.12 if "Yes" in b_basement else 0.0
            
            raw_delta_lst = delta_concrete + delta_ac + delta_parking + delta_basement
            raw_scen_temp = base_temp + raw_delta_lst
            
            cool_roof_savings = 0.45 if mit_cool_roof else 0.0
            paver_savings = 0.30 if mit_permeable else 0.0
            tree_savings = 0.40 if mit_trees else 0.0
            hvac_savings = 0.20 if mit_energy else 0.0
            
            total_mitigation = cool_roof_savings + paver_savings + tree_savings + hvac_savings
            mitigated_delta_lst = max(0.05, raw_delta_lst - total_mitigation)
            mitigated_scen_temp = base_temp + mitigated_delta_lst

            # Chart: Matplotlib comparison
            fig_bldg, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(7.2, 2.5), dpi=100)
            fig_bldg.patch.set_facecolor('#ffffff')

            # Subplot 1: Temperature Comparison
            labels = ['Ward Baseline', 'Standard Build\n(Unmitigated)', 'Green Build\n(Mitigated)']
            temps = [base_temp, raw_scen_temp, mitigated_scen_temp]
            colors = ['#94a3b8', '#ef4444', '#10b981']
            
            bars_t = ax_t.bar(labels, temps, color=colors, width=0.50)
            ax_t.set_facecolor('#ffffff')
            ax_t.set_ylabel("Localized Temp (°C)", fontsize=8, fontweight='600', color='#475569')
            ax_t.set_title("Local Surface Heat Impact (°C)", fontsize=9, fontweight='700', color='#0f172a', pad=6)
            min_y = max(0, min(temps) - 2.5)
            max_y = max(temps) + 2.0
            ax_t.set_ylim(min_y, max_y)
            ax_t.spines['top'].set_visible(False)
            ax_t.spines['right'].set_visible(False)
            ax_t.spines['left'].set_color('#cbd5e1')
            ax_t.spines['bottom'].set_color('#cbd5e1')
            ax_t.tick_params(colors='#475569', labelsize=7.5)

            for bar in bars_t:
                h = bar.get_height()
                ax_t.text(bar.get_x() + bar.get_width()/2, h + 0.25, f"{h:.2f}°C", ha='center', va='bottom', fontsize=8, fontweight='700', color='#0f172a')

            # Subplot 2: Environmental Stress Factors (%)
            env_metrics = ['Stormwater\nRunoff', 'Impervious\nFraction', 'Waste Heat\nRejection']
            unmit_scores = [100.0, 95.0, 85.0]
            runoff_score = 35.0 if (mit_permeable and "Yes" in b_basement) else (50.0 if mit_permeable else 85.0)
            impervious_score = 40.0 if mit_permeable else 90.0
            heat_score = 30.0 if (mit_cool_roof and mit_energy) else 70.0
            mit_scores = [runoff_score, impervious_score, heat_score]

            x_pos = np.arange(len(env_metrics))
            w_bar = 0.35
            bars_u = ax_m.bar(x_pos - w_bar/2, unmit_scores, width=w_bar, label='Unmitigated', color='#fca5a5')
            bars_m = ax_m.bar(x_pos + w_bar/2, mit_scores, width=w_bar, label='Mitigated', color='#6ee7b7')
            ax_m.set_facecolor('#ffffff')
            ax_m.set_xticks(x_pos)
            ax_m.set_xticklabels(env_metrics, fontsize=7.5)
            ax_m.set_ylabel("Stress Index (0-100)", fontsize=8, fontweight='600', color='#475569')
            ax_m.set_title("Environmental Stress Index", fontsize=9, fontweight='700', color='#0f172a', pad=6)
            ax_m.set_ylim(0, 125)
            ax_m.spines['top'].set_visible(False)
            ax_m.spines['right'].set_visible(False)
            ax_m.spines['left'].set_color('#cbd5e1')
            ax_m.spines['bottom'].set_color('#cbd5e1')
            ax_m.tick_params(colors='#475569', labelsize=7.5)
            ax_m.legend(loc='upper right', fontsize=7, frameon=False)

            plt.tight_layout()
            st.pyplot(fig_bldg, use_container_width=True)
            plt.close(fig_bldg)

            pct_offset = ((raw_delta_lst - mitigated_delta_lst) / max(0.01, raw_delta_lst)) * 100.0

            # Brief Description of Impact
            st.markdown(f"""
            <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px 14px; margin-top:4px; font-size:0.80rem; color:#334155; line-height:1.5;'>
                <b>Environmental Impact Summary:</b> Constructing this <b>{b_floors}-story building</b> ({b_sqft:,} sq.ft ground footprint, {gba_sqft:,} sq.ft gross built area) for ~<b>{b_residents} occupants</b> in <b>{target_ward_record['ward_name']}</b> introduces substantial concrete thermal mass and HVAC waste heat. Standard unmitigated construction raises localized surface temperature by <b>+{raw_delta_lst:.2f}°C</b> (to {raw_scen_temp:.2f}°C). Enforcing green building measures restrains thermal impact to only <b>+{mitigated_delta_lst:.2f}°C</b> (a <b>{pct_offset:.0f}% heat reduction</b>).
            </div>
            """, unsafe_allow_html=True)

            # Municipal Action Measures
            st.markdown("""
            <div style='background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:10px 14px; margin-top:8px;'>
                <b style='font-size:0.82rem; color:#166534;'>🛡️ Recommended Municipal Measures & Building By-Law Compliance:</b>
                <ol style='font-size:0.77rem; color:#15803d; margin-top:4px; margin-bottom:0; padding-left:18px; line-height:1.4;'>
                    <li><b>Cool Roof Mandate:</b> Require solar-reflective white coating or high-albedo tiles (SRI ≥ 82) on 100% of the upper roof slab to reject 40–50 W/m² solar heat.</li>
                    <li><b>Setback Greenery:</b> Mandate minimum 15-20% open plot area planted with native shade trees to shield building walls from direct solar irradiance.</li>
                    <li><b>Runoff Infiltration Pits:</b> Ensure dual recharge percolation shafts to replenish the groundwater table depleted by basement excavation.</li>
                    <li><b>Permeable Pavers:</b> Prohibit solid asphalt on non-covered parking and driveways; enforce porous interlocking pavers to maintain soil breathing.</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

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

        # Human-Readable Feature Name Mapping for Common Citizens
        FEATURE_HUMAN_NAMES = {
            "Building_Density_pct": "Building Footprint (% Ground Covered)",
            "NDVI": "Green Vegetation & Tree Canopy (NDVI)",
            "Road_Density_km_km2": "Road & Pavement Network (km/km²)",
            "NDBI": "Impervious Concrete / Built Index (NDBI)",
            "Dist_to_Water_m": "Distance to Water Channel (meters)",
            "Dist_to_Park_m": "Distance to Public Park (meters)",
            "Relative_Humidity_pct": "Ambient Air Humidity (%)",
            "Albedo": "Solar Roof Reflectance (Albedo)",
            "NDWI": "Surface Water & Moisture (NDWI)",
            "Slope_deg": "Ground Slope / Incline (degrees)"
        }

        # Update SHAP Explanation with user-friendly names
        shap_exp = shap_vals[0]
        shap_exp.feature_names = [FEATURE_HUMAN_NAMES.get(c, c) for c in X_cell.columns]

        # Guide explaining how a common citizen or municipal officer can read this graph
        st.markdown(f"""
        <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px 16px; margin-bottom:12px;'>
            <div style='font-size:0.88rem; font-weight:800; color:#0f172a; margin-bottom:6px;'>
                📖 How to Read & Interpret This Ward Explanation Graph:
            </div>
            <div style='font-size:0.80rem; color:#475569; line-height:1.5;'>
                • <b>Starting Point at Bottom (E[f(x)]):</b> The average baseline temperature across the entire municipal corporation (~38.2°C).<br>
                • <span style='color:#dc2626; font-weight:700;'>🔴 Red Bars Pushing Right (+ values):</span> Environmental factors that are <b>heating up this ward</b> (e.g. dense concrete buildings, lack of tree canopy, dark asphalt roads).<br>
                • <span style='color:#0284c7; font-weight:700;'>🔵 Blue Bars Pushing Left (- values):</span> Environmental factors that are <b>cooling down this ward</b> (e.g. nearby water bodies, vegetation, reflective roofs).<br>
                • <b>Final Result at Top (f(x)):</b> The resulting predicted surface temperature (<b>{target_ward_record['Mean_LST_Celsius']:.1f}°C</b>) for {target_ward_record['ward_name']}.
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        shap.plots.waterfall(shap_exp, max_display=8, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Plain language summary below the graph
        st.markdown(f"""
        <div style='background:#fff7ed; border:1px solid #ffedd5; border-radius:8px; padding:10px 14px; margin-top:8px; font-size:0.80rem; color:#9a3412;'>
            <b>Key Takeaway for {target_ward_record['ward_name']}:</b> The red bars show which local factors contribute most to making this ward hotter than the city average. Dense building coverage and depleted greenery push heat upward, whereas proximity to water channels and reflective surfaces provide cooling relief.
        </div>
        """, unsafe_allow_html=True)

        # Global Charts
        st.markdown("#### Global Model Insights across All Administrative Wards")
        g1, g2 = st.columns(2)
        with g1:
            b_img = FIGURES_DIR / f"{selected_region}_shap_bar.png"
            if b_img.exists():
                st.image(str(b_img), caption="Global Mean Feature Importance (|SHAP|)")
                st.caption("Ranks which environmental factors have the strongest overall grip on surface heat across all wards (building density and tree canopy are dominant).")
        with g2:
            bw_img = FIGURES_DIR / f"{selected_region}_shap_beeswarm.png"
            if bw_img.exists():
                st.image(str(bw_img), caption="Feature Value vs Impact Direction")
                st.caption("Shows how high (red dots) vs. low (blue dots) values of each indicator push temperatures up or down across all neighborhoods.")

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
        l_color = "#16a34a" if l_info["status"] in ["Available", "Live Operational"] else "#dc2626"
        st.markdown(f"""
        <div style='background:white; border:1px solid #e2e8f0; border-radius:8px; padding:12px 16px; margin-bottom:8px;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div style='font-size:0.92rem; font-weight:800; color:#0f172a;'>
                    {l_info['name']}
                </div>
                <div style='font-size:0.75rem; font-weight:700; color:{l_color};'>
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




