# Urban Heat Vulnerability & Risk Assessment: Project Walkthrough

## Executive Summary
The **Explainable Machine Learning Urban Heat Vulnerability Assessment and Risk Mapping Platform (Municipal Heat Watch)** is fully operational. It establishes an active early warning and decision support system comparing the **Cauvery Delta urban fringe (Tiruchirappalli)** with a **dense coastal metropolis (Chennai)**.

The framework complies with **National Disaster Management Authority (NDMA)** guidelines and municipal smart city specifications:
1. **Decoupled Risk Modeling:** Physical Land Surface Temperature (LST exposure) is modeled via machine learning independently from demographic vulnerability, producing a composite **Heat Risk Index ($\text{Risk} = \sqrt{\text{Exposure} \times \text{Vulnerability}}$)**.
2. **Administrative Jurisdictions:** Predictions from 9,676 micro-grid cells across both cities are aggregated to official **Municipal Wards**, yielding a **Ward Heat Risk Index (WHRI)** and priority ranking (Rank 1 to $N$) with specific NDMA-aligned Heat Action Plan interventions.
3. **Mission-Control SaaS Interface:** Features a high-contrast top-navigation dashboard with live emergency action dispatchers, interactive Folium maps with free basemaps, and real-time ML inference.
4. **Explainable AI (SHAP):** Features both global non-linear threshold discovery (NDVI cooling tipping points) and **real-time on-the-fly local SHAP Waterfall explanations** (<10ms).

---

## Machine Learning & Spatial Cross-Validation Benchmarks

To prevent spatial autocorrelation data leakage, models were evaluated using **Spatial Block K-Fold Cross-Validation** (5 geographic block splits).

### 1. Tiruchirappalli (Cauvery Delta Gateway - 5,040 Grid Cells, 65 Wards)
| Model | Spatial CV $R^2$ Mean (± Std) | Spatial CV RMSE (°C) | Spatial CV MAE (°C) |
| :--- | :---: | :---: | :---: |
| **Ridge Regression (Baseline, L2 alpha=1.0)** | **0.8765 (± 0.0505)** | **0.821** | **0.652** |
| **Random Forest Regressor** | 0.8689 (± 0.0568) | 0.839 | 0.666 |
| **XGBoost Regressor** | 0.8687 (± 0.0562) | 0.842 | 0.670 |

* **PCA Analysis:** Top-3 principal components capture **64.06%** of spatial variance ($PC_1$: 40.99%, $PC_2$: 14.58%, $PC_3$: 8.49%).

### 2. Chennai Metropolitan Area (Non-Delta Coastal Metro - 4,636 Grid Cells, 100 Wards)
| Model | Spatial CV $R^2$ Mean (± Std) | Spatial CV RMSE (°C) | Spatial CV MAE (°C) |
| :--- | :---: | :---: | :---: |
| **XGBoost Regressor (Best)** | **0.9502 (± 0.0558)** | **0.731** | **0.511** |
| **Random Forest Regressor** | 0.9438 (± 0.0611) | 0.785 | 0.532 |
| **Ridge Regression (Baseline, L2 alpha=1.0)** | 0.8985 (± 0.0730) | 1.207 | 0.945 |

* **PCA Analysis:** Top-3 principal components capture **71.19%** of spatial variance ($PC_1$: 35.79%, $PC_2$: 26.69%, $PC_3$: 8.71%).

---

## Administrative Heat Action Plan (HAP) Priority Rankings

### Top 5 Emergency Priority Wards in Tiruchirappalli (Delta)
| Rank | Ward ID | Ward Name | Mean LST (°C) | Ward Heat Risk Index (WHRI) | Administrative Tier |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | `W_032` | Tiruch Ward 32 | **44.65°C** | **0.767** | **Extreme Priority** |
| **#2** | `W_033` | Tiruch Ward 33 | **44.24°C** | **0.704** | **Extreme Priority** |
| **#3** | `W_023` | Tiruch Ward 23 | **42.28°C** | **0.615** | **Extreme Priority** |
| **#4** | `W_024` | Tiruch Ward 24 | **42.15°C** | **0.604** | **Extreme Priority** |
| **#5** | `W_041` | Tiruch Ward 41 | **42.03°C** | **0.577** | **Extreme Priority** |

### Top 5 Emergency Priority Wards in Chennai (Non-Delta)
| Rank | Ward ID | Ward Name | Mean LST (°C) | Ward Heat Risk Index (WHRI) | Administrative Tier |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | `W_057` | Chenna Ward 57 | **44.97°C** | **0.831** | **Extreme Priority** |
| **#2** | `W_067` | Chenna Ward 67 | **45.00°C** | **0.830** | **Extreme Priority** |
| **#3** | `W_056` | Chenna Ward 56 | **44.88°C** | **0.774** | **Extreme Priority** |
| **#4** | `W_058` | Chenna Ward 58 | **44.79°C** | **0.769** | **Extreme Priority** |
| **#5** | `W_068` | Chenna Ward 68 | **44.85°C** | **0.760** | **Extreme Priority** |

---

## Key System Modules in the Redesigned Platform

1. **Executive Summary & Action Dispatcher Console:**
   - Real-time ward sorting table with status indicators (`● Dispatching`, `Pending Review`, `Shelters Open`, `Hydration Active`, `Monitored`).
   - Two diagnostic sub-cards: `☀️ Delta Riverbed Heat Sinks` and `➕ District Hospital Alert`.
   - Dedicated Emergency Action Dispatcher with `LEVEL 1 HOTSPOT` tag, vulnerability score (`94/100`), and 4 operational protocol cards (`💧 Water & Hydration Kiosks`, `🏥 Primary Health Centre Surge`, `🏠 Albedo Cool-Roof Spray`, `👷 Outdoor Labor Work Stoppage`).
   - Action triggers: `📢 Dispatch Municipal Directives for Ward 32`, `📄 Generate PDF Order`, and `🗺️ View on Spatial Map`.
2. **Spatial Heat Map (Zero Watermark):**
   - Free Leaflet choropleth integration with seamless toggle between **OpenStreetMap** and **Esri World Imagery Satellite**.
   - Interactive ward boundary tooltips and color-coded risk legends.
3. **Real-Time Area Inspector (Live ML & SHAP on the fly):**
   - Live microclimate parameter adjusters (NDVI, NDBI, Building Density, Distance to Water).
   - Instantaneous prediction of surface heat and composite risk index.
   - On-the-fly **SHAP Waterfall breakdown** showing degree-by-degree temperature attribution (e.g., $+4.2^\circ\text{C}$ from building density, $-0.41^\circ\text{C}$ from low NDBI).
4. **Explainable AI (SHAP) Global Insights:**
   - Global beeswarm and feature importance rankings.
   - Non-linear tipping points: **$\text{NDVI} > 0.32$** delivers exponential temperature drops of **$2.5^\circ\text{C} - 4.0^\circ\text{C}$**.
   - Riparian envelope: Proximity to the Cauvery river system creates a **$1.8^\circ\text{C} - 2.5^\circ\text{C}$** cooling buffer within 500m.
5. **Greening Studio:**
   - Simulates urban tree canopy expansion (+NDVI) and cool roof coatings (-NDBI) to forecast temperature drops before committing municipal budgets.
6. **Live API Ingestion & Future Roadmap:**
   - Real-time weather streaming from Open-Meteo API.
   - Step-by-step guidance on external weather API ingestion and rate-limit caching.
   - 3-tier extension roadmap: Freehand polygon drawing, 1-click automated PDF reports, and grid peak AC load forecasting.

---

## Launch Instructions

To launch the platform in your browser at `http://localhost:8501`:

```powershell
.\run_dashboard.bat
```
*(Or in PowerShell)*:
```powershell
.\run_dashboard.ps1
```
*(Or via Python module syntax)*:
```powershell
python -m streamlit run app/dashboard.py
```
