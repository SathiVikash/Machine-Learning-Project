# Implementation Plan: Explainable ML for Urban Heat Vulnerability Assessment & Risk Mapping

## Project Status: Completed & Operational

This document records the architectural design, methodology, validation benchmarks, and UI engineering for the **Explainable Machine Learning Urban Heat Vulnerability and Early Warning Decision Support System (Municipal Heat Watch)**, comparing the **Cauvery Delta urban-agricultural fringe (Tiruchirappalli)** with a **dense coastal metropolis (Chennai)**.

---

## 1. Core Methodological Foundation

### Decoupled Risk Architecture
To avoid the circular logic of training a model on an artificially computed index, the framework enforces a three-stage decoupled architecture:

$$\text{Stage 1: Physical Exposure Modeling} \implies \text{LST} = f(\text{Spectral}, \text{Terrain}, \text{Urban Form}, \text{Climate})$$
$$\text{Stage 2: Socio-Spatial Vulnerability} \implies V = \sum w_i \cdot \text{Norm}(\text{Demographic Indicators})$$
$$\text{Stage 3: Composite Heat Risk Index} \implies \text{HRI} = \sqrt{\text{Exposure Score} \times \text{Vulnerability Score}}$$

### Spatial Unit Architecture
- **Under-the-Hood Modeling Unit:** Regular analytical grid ($250\text{m} \times 250\text{m}$ fishnet) capturing street-scale microclimatic variations.
- **Administrative Governance Unit:** Official **Municipal Corporation Wards**, providing municipal commissioners and disaster management authorities with actionable **Ward Heat Risk Index (WHRI)** rankings and Heat Action Plan (HAP) mandates.

---

## 2. Completed Implementation Phases

```
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Data Ingestion & Spatial Harmonization (COMPLETED)           │
│ - 5,040 grid cells & 65 wards generated for Tiruchirappalli (Delta)    │
│ - 4,636 grid cells & 100 wards generated for Chennai (Non-Delta)       │
│ - Harmonized Landsat LST, Sentinel-2 (NDVI, NDBI, NDWI), DEM, & OSM    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Spatial Machine Learning & Validation (COMPLETED)             │
│ - 5-Fold Spatial Block Cross-Validation (prevents spatial leakage)     │
│ - Models: Ridge (Baseline), Random Forest, XGBoost Regressor           │
│ - PCA Multicollinearity & Variance Analysis                            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Explainable AI & Factor Attribution (COMPLETED)               │
│ - Global TreeExplainer: Beeswarm plots & feature importance bar charts │
│ - Non-linear tipping points: NDVI cooling curve & NDBI heating curve   │
│ - Local real-time SHAP Waterfall explainer (<10ms inference)           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Municipal Heat Watch Platform (COMPLETED)                     │
│ - Pixel-perfect mission-control UI matching NDMA / HAP guidelines      │
│ - Real-Time Area Inspector with live feature sliders                   │
│ - Emergency Action Dispatcher with 4-tier mandated protocol cards      │
│ - Live Weather API Ingestion via Open-Meteo & future extension roadmap │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Validation & Performance Benchmarks

### Spatial Block Cross-Validation (5 Geographic Folds)

| Region | Best Performing Model | Spatial CV $R^2$ Mean (± Std) | RMSE (°C) | MAE (°C) | PCA Top-3 Var (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Tiruchirappalli (Delta)** | Linear Baseline / RF | **0.8765 (± 0.0505)** | 0.821 | 0.652 | 64.06% |
| **Chennai (Non-Delta)** | XGBoost Regressor | **0.9502 (± 0.0558)** | 0.731 | 0.511 | 71.19% |

### Key Physical & Explainability Findings
1. **NDVI Non-Linear Cooling Threshold:** Vegetation cooling accelerates exponentially when $\text{NDVI} > 0.32$, reducing surface temperatures by **$2.5^\circ\text{C}$ to $4.0^\circ\text{C}$**.
2. **Delta Riparian Envelope:** Proximity to the Cauvery/Kollidam river channels delivers a measured **$1.8^\circ\text{C} - 2.5^\circ\text{C}$** cooling buffer within a **500-meter** envelope.
3. **Impervious Surface Tipping Point:** Building density exceeding **65%** and $\text{NDBI} > 0.25$ triggers rapid heat accumulation due to nocturnal thermal storage and lack of evaporative cooling.

---

## 4. UI/UX Architecture & System Features

1. **Top Brand Header Bar:**
   - Brand logo with flame icon, `MUNICIPAL HEAT WATCH`, and `● Live Telemetry` badge.
   - Dual-city case study switcher (Cauvery Delta vs. Chennai Coastal Metro).
   - Dynamic alert status badge (`⚠️ Heat Alert: Level 2 Orange`), model timestamp, and `📥 Export Report` button.
2. **Segmented Horizontal Navbar:**
   - Active view highlighted in solid deep slate (`#0f172a`) with white typography.
   - Modules: *Executive Summary*, *Spatial Heat Map*, *Real-Time Area Inspector*, *Explainable AI (SHAP)*, *Greening Studio*, *Delta vs Non-Delta*, *Live API & Future Extensions*.
3. **5-Deck KPI Deck:**
   - Live metrics for Total Wards, Mean Surface Temp, Emergency Wards, Spatial Model $R^2$, and Population at Risk.
4. **Emergency Action Dispatcher Console:**
   - Real-time ward selector displaying `LEVEL 1 HOTSPOT` tag, vulnerability score (e.g., `94/100`), and 4 operational protocol cards:
     - `💧 Water & Hydration Kiosks` (Active Deployment)
     - `🏥 Primary Health Centre (PHC) Surge` (Clinics Alerted)
     - `🏠 Albedo Cool-Roof Spray` (In Progress)
     - `👷 Outdoor Labor Work Stoppage` (Mandate Enforced)
   - Primary `📢 Dispatch Municipal Directives` button and secondary PDF/Map triggers.
5. **Real-Time Area Inspector:**
   - Real-time feature sliders (NDVI, NDBI, Building Density, Water Distance).
   - Sub-10ms ML inference recalculating predicted LST and risk tier.
   - On-the-fly **SHAP Waterfall Plot** showing exact feature-by-feature degree contributions.
6. **Live Weather API Ingestion & Future Extensions:**
   - Real-time atmospheric streaming from Open-Meteo API.
   - Clear architectural guidance on feeding live ambient streams into ML feature inputs.
   - 3-tier extension roadmap: Leaflet-Draw freehand zoning, 1-click PDF municipal order generation, and electrical grid peak AC load forecasting.
