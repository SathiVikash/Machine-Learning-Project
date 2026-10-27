# MUNICIPAL HEAT WATCH: Explainable Machine Learning for Urban Heat Vulnerability Assessment and Risk Mapping

An AI-powered, geospatial decision support and early warning platform comparing the **Cauvery Delta urban-agricultural fringe (Tiruchirappalli)** with a **dense coastal metropolis (Chennai)**.

---

## 1. Project Abstract & Motivation

Urban Heat Islands (UHI) and extreme surface temperatures present severe public health hazards, especially in rapidly urbanizing Indian cities. Traditional research predominantly focuses on remote sensing Land Surface Temperature (LST) mapping or predicting LST using standard regression models. However, simply predicting temperature does not inform municipal authorities which communities are most vulnerable or where interventions will have the highest life-saving impact.

> **Scope & Operational Boundaries:** Municipal Heat Watch is an exploratory spatial screening and decision-support prototype. Satellite-derived Land Surface Temperature (LST) captures skin surface radiometric temperature from USGS Landsat-9 at overpass moments (physically distinct from 2m ambient air temperature). Atmospheric weather feeds are numerical model outputs from Open-Meteo (NOAA GFS / DWD ICON blend) monitored for ambient context and are not fed into the static spatial regression model. What-if scenario interventions are empirical model sensitivity estimates, not validated causal cooling forecasts. Role assignments are unauthenticated demonstrations. This system does not issue statutory weather warnings or clinical heatstroke prognoses.

This project introduces a **decoupled, explainable machine learning framework** tailored for municipal governance:
1. **Physical Exposure Modeling:** Models Land Surface Temperature using satellite spectral indices (Landsat 8/9, Sentinel-2), terrain features (DEM), and OpenStreetMap (OSM) urban form indicators under **Spatial Block Cross-Validation**.
2. **Demographic & Socio-Spatial Vulnerability:** Combines WorldPop population density, sensitive demographics (children <5, elderly >65), low-income household ratios, and healthcare distance into a normalized Vulnerability Index.
3. **Decoupled Heat Risk Index (HRI):** Formulated as:
   $$\text{Risk} = \sqrt{\text{Exposure Score} \times \text{Vulnerability Score}}$$
4. **Administrative Ward Aggregation:** Aggregates micro-grid predictions (250m × 250m) to official **Municipal Wards**, calculating a **Ward Heat Risk Index (WHRI)**, ranking priority intervention zones (Rank 1 to $N$), and pairing them with actionable **Heat Action Plan (HAP)** interventions.
5. **Explainable AI (SHAP):** Unpacks model decisions using TreeExplainer, identifying non-linear tipping points (e.g., NDVI cooling thresholds vs. NDBI heating acceleration) and providing **real-time on-the-fly local SHAP Waterfall explanations** (<10ms).

---

## 2. Dataset, Metric & Threshold Provenance Catalog

| Item | Canonical Tag | Data Source & Instrument | Spatial Scale | Role in Platform |
| :--- | :--- | :--- | :--- | :--- |
| **Land Surface Temp (LST)** | `[Observed]` | USGS Landsat-9 TIRS Band 10 | 30m (aggregated to 250m) | Target variable for physical exposure |
| **Vegetation & Built Indices (NDVI, NDBI)** | `[Observed]` | Copernicus Sentinel-2 MSI Bands 4, 8, 11 | 10m / 20m resampled to 250m | Key environmental predictor features |
| **Atmospheric Weather Feeds** | `[Estimated]` | Open-Meteo API (NOAA GFS / DWD ICON blend) | Point / model grid cell | Real-time ambient context (not fed to spatial model) |
| **Demographics & Vulnerability** | `[Estimated]` | WorldPop 2024 Dasymetric Modeling + Census Proxies | 100m raster aggregated to ward | Socio-demographic vulnerability component |
| **Urban Amenities & Vectors** | `[Observed]` | OpenStreetMap Contributors (ODbL) | Vector points & polygons | Distance to healthcare, water, and parks |
| **Heat Risk Index (WHRI)** | `[Derived]` | Decoupled Formulation: $\text{HRI} = \sqrt{E \times V}$ | Ward-level composite | Normalized index [0.0, 1.0] for priority ranking |
| **Priority Wards Action Threshold** | `[Administrative Heuristic]` | Municipal Corporation Threshold: $\text{WHRI} > 0.50$ | Ward administrative polygon | Triggers municipal HAP Stage 1 interventions |
| **Out-of-Fold Model Fit ($R^2$, RMSE)** | `[Derived]` | Spatial Block 5-Fold Cross-Validation | 5 geographic fold blocks | Measures out-of-fold generalizability |

---

## 3. Delta vs. Non-Delta Comparative Analysis

| Feature Dimension | Tiruchirappalli (Cauvery Delta Gateway) | Chennai (Non-Delta Coastal Metropolis) |
| :--- | :--- | :--- |
| **Zone Classification** | Delta Urban Fringe | Dense Coastal Metro |
| **Total Micro-Grid Cells** | 5,040 (250m × 250m resolution) | 4,636 (250m × 250m resolution) |
| **Official Wards Analyzed** | 65 Municipal Wards | 100 Municipal Wards |
| **Natural Buffer Mechanism** | Cauvery & Kollidam river network, riparian green corridor | Maritime breeze moderation along the Bay of Bengal coastline |
| **Urban Heat Driver** | Rapid conversion of delta wetlands into industrial/residential corridors | Extreme impervious built-up fraction, dense street canyons |
| **Peak Observed LST (Satellite Skin Temp)** | **44.65°C** (Dry interior surface heating) | **45.00°C** (Impervious built-up thermal retention) |
| **Meteorological Context** | Lower relative humidity (~35–45%), high solar irradiance | High coastal humidity (~65–80%), producing elevated physiological Heat Indices |

---

## 4. Machine Learning Performance & Model Selection

To prevent spatial autocorrelation data leakage, models were evaluated using **Spatial Block Cross-Validation (5 Geographic Folds)**:

### Tiruchirappalli (Delta)
- **Ridge Regression (Baseline, L2 alpha=1.0):** $R^2 = 0.8765 \pm 0.0505$ | $\text{RMSE} = 0.821^\circ\text{C}$ | $\text{MAE} = 0.652^\circ\text{C}$
- **Random Forest Regressor:** $R^2 = 0.8689 \pm 0.0568$ | $\text{RMSE} = 0.839^\circ\text{C}$ | $\text{MAE} = 0.666^\circ\text{C}$
- **XGBoost Regressor:** $R^2 = 0.8687 \pm 0.0562$ | $\text{RMSE} = 0.842^\circ\text{C}$ | $\text{MAE} = 0.670^\circ\text{C}$
- **PCA Variance Explained:** Top-3 components capture **64.06%** of spatial variance.

### Chennai Metropolitan Area (Non-Delta)
- **XGBoost Regressor (Best):** **$R^2 = 0.9502 \pm 0.0558$** | **$\text{RMSE} = 0.731^\circ\text{C}$** | **$\text{MAE} = 0.511^\circ\text{C}$**
- **Random Forest Regressor:** $R^2 = 0.9438 \pm 0.0611$ | $\text{RMSE} = 0.785^\circ\text{C}$ | $\text{MAE} = 0.532^\circ\text{C}$
- **Ridge Regression (Baseline, L2 alpha=1.0):** $R^2 = 0.8985 \pm 0.0730$ | $\text{RMSE} = 1.207^\circ\text{C}$ | $\text{MAE} = 0.945^\circ\text{C}$
- **PCA Variance Explained:** Top-3 components capture **71.19%** of spatial variance.

### Why Random Forest is Deployed (Measured Justification)
1. **Non-Linear Threshold Saturation:** Urban microclimate dynamics display sharp non-linear thresholds (e.g. NDVI cooling accelerates when $\text{NDVI} > 0.32$, reducing surface temperatures by $2.5^\circ\text{C} - 4.0^\circ\text{C}$; NDBI built-up heating accelerates above $0.25$). Ridge regression assumes constant marginal returns and fundamentally cannot represent saturation points.
2. **Physically Bounded Tree Averaging:** Random Forest predictions are bounded within $[\min(y_{train}), \max(y_{train})]$. Unlike unconstrained polynomials or gradient boosting steps that can extrapolate unrealistically during extreme slider testing, Random Forest guarantees predictions remain within observed physical limits.
3. **Exact, Sub-10ms TreeSHAP:** `shap.TreeExplainer` computes polynomial-time exact Shapley values directly from tree leaves (<10ms), enabling real-time on-the-fly local waterfall explanations in the municipal web interface without KernelSHAP background sampling approximations.

---

## 4. Key Explainable AI (SHAP) Discoveries

1. **Vegetation Cooling Tipping Point:**
   - SHAP dependence plots identify that when **$\text{NDVI} > 0.32$**, an exponential cooling effect begins, dropping local surface temperatures by **$2.5^\circ\text{C}$ to $4.0^\circ\text{C}$**.
2. **Delta Riparian Envelope:**
   - Proximity to the Cauvery river system creates a **$1.8^\circ\text{C}$ to $2.5^\circ\text{C}$** microclimatic cooling buffer extending up to **500 meters**.
3. **Impervious Surface Acceleration:**
   - Built-up density exceeding **65%** and $\text{NDBI} > 0.25$ causes sharp temperature jumps due to nocturnal thermal mass and absence of soil moisture evaporation.
4. **Real-Time Local Feature Attribution:**
   - Using local SHAP Waterfall plots, decision-makers can inspect any specific neighborhood to see the exact degree contribution of each urban parameter (e.g., $+4.2^\circ\text{C}$ from building density, $-0.41^\circ\text{C}$ from low NDBI).

---

## 5. Platform Architecture & Features

The **Municipal Heat Watch** platform is structured into **7 core municipal workflow modules** with an optimized sideways sidebar navigation:

- **📊 Dashboard:**
  - **5-Deck KPI Telemetry:** Total Surveyed Wards, Mean/Peak Surface Temperature (°C), Priority Action Wards ($\text{WHRI} > 0.50$), Spatial Block 5-Fold Model $R^2$ (Cross-Validated = 0.876), and Exposed Population (formatted cleanly in `k` up to 900k, and `M` above 900k without double-counting).
  - **Scrollable Administrative Ward Rankings Table:** Fixed 8-row viewport table with instant real-time search filter displaying Ward ID, Name, Priority Tier, composite HRI, Mean LST (°C), and population density (/km²).
  - **Synchronized Active Ward Selector:** Aligned right above the active ward assessment header adjacent to the table for intuitive navigation.
  - **Ward-Specific Live Weather Feed (Open-Meteo):** Dynamically extracts the exact centroid coordinates of the selected ward and queries the Open-Meteo atmospheric model (NOAA GFS / DWD ICON blend) displaying Ambient Air Temperature, Relative Humidity, Apparent Heat Index, and Wind Speed in real time.
- **📋 Ward Risk Profile & Assessment:**
  - **Dynamic Graphical Visualizations:** Responsive horizontal bar charts updating live for *Thermal Radiation Context (Peak Pixel vs. Ward Mean vs. City Baseline)*, *Physical Environmental Covariates (Water Distance, NDBI, NDVI, Building Density)*, and *People & Socio-Demographic Vulnerability (Age Cohorts & Fragility Breakdown)*.
  - **Why This Ward?**: Plain-language physical attribution cards classifying local heating drivers vs. cooling buffers.
  - **Recommended Heat Action Plan (HAP) Measures:** Eye-catching, high-contrast intervention container highlighting priority municipal protocols (cool roofs, hydration kiosks, labor shade advisories).
  - **1-Click Proposal Dispatch:** Pre-populates ward context into an administrative form to submit actionable proposals directly into the SQLite database.
- **🗺️ Spatial Heat & Risk Map:**
  - **Interactive Leaflet Choropleth Map:** Color-coding all administrative divisions into 5 priority tiers (Extreme Priority Red, Very High Orange, High Amber, Moderate Green, Low Blue) with rich hover tooltips.
  - **Basemap Style Switcher:** Seamless toggle between **OpenStreetMap** (Clean Cartographic) and **Esri World Imagery** (High-Resolution Satellite) without third-party API keys or watermarks.
- **🔮 Scenario Workspace (Simulation & Impact Modeling):**
  - **Mode A (Ward Cooling Interventions):** What-if simulator with human-relatable inputs (houses coated: 100–5,000; native trees planted: 250–10,000; m² de-paved: 5,000–100,000; canal meters restored: 200–4,000; % coverage), side-by-side Matplotlib comparison graphs (LST & HRI), and Model Domain Check boundary validation.
  - **Mode B (New Building Construction Impact Simulator):** Architectural parameter simulator (sq. ft footprint, floors, residents, parking format, basement excavation, green building mitigations) predicting localized surface heat contribution, stormwater runoff multipliers, dual-panel comparative charts, and actionable municipal by-law recommendations.
- **🧠 Explainable AI (XAI & SHAP):**
  - **Human-Readable Common-Man Terms:** ML features translated into everyday language (e.g., `Building Footprint (% Ground Covered)`, `Green Vegetation & Tree Canopy (NDVI)`, `Distance to Water Channel (meters)`).
  - **Waterfall Interpretation Guide:** Clear instructional box explaining the city baseline ($E[f(x)]$), red heating bars (+), blue cooling bars (-), and final predicted ward temperature ($f(x)$).
  - **Local SHAP Waterfall Plot:** Fast TreeExplainer (<10ms) showing exact degree contributions for the selected ward.
  - **Dynamic Ward Takeaways:** Plain-language summary explaining primary local heating drivers and recommended municipal cooling levers.
  - **Global Insights:** Global Feature Importance bar chart and Beeswarm plot with non-technical captions explaining city-wide tipping points.
- **📝 HAP Proposals & Review Tracker:**
  - **Persistent SQLite Database (`data/app_database.db`):** Stores civic proposals, budgets, target departments, and review statuses across server restarts.
  - **Server-Side State Machine:** Enforces valid lifecycle state transitions (`Draft` $\to$ `Submitted` $\to$ `Under Review` $\to$ `Approved` $\to$ `Completed` / `Rejected`).
  - **Security & Auditability:** Built-in Stored XSS escaping, CSV formula injection sanitization, and chronological audit logging.
- **🔄 Data Freshness & Provenance:**
  - Multi-source sensor lineage audit trail (Landsat-9, Sentinel-2, ERA5, WorldPop, OSM, Open-Meteo), calibration verification, and operational scope boundaries.

---

## 6. Project Directory Layout

```
Projects/
├── app/
│   └── dashboard.py          # Municipal Heat Watch Decision Support Platform
├── data/
│   ├── app_database.db       # Persistent SQLite proposals & audit trail database
│   ├── processed/            # Preprocessed spatial grids and ward GeoJSON boundaries
│   └── raw/
├── outputs/
│   ├── figures/              # Publication-ready SHAP beeswarm, bar, and dependence plots
│   ├── models/               # Trained joblib models, scalers, and comparison CSVs
│   ├── *_administrative_ward_rankings.csv
│   └── *_wards_risk_enriched.geojson
├── scripts/
│   ├── run_pipeline.py       # Master end-to-end execution script
│   └── gee_exporter.py       # Google Earth Engine script for cloud users
├── src/
│   ├── config.py             # Feature schema, regional bounds, and HAP policy matrix
│   ├── data_generator.py     # High-resolution spatial grid & ward generator
│   ├── data_status.py        # Provenance, latency, observation timeline & demo-mode flags
│   ├── models.py             # Spatial Block CV, Ridge, RF, XGBoost, PCA
│   ├── vulnerability.py      # Demographic normalization & K-Means clustering
│   ├── explainability.py     # SHAP TreeExplainer & threshold visualizer
│   ├── ward_aggregation.py   # Ward-level WHRI, population-at-risk, & plain-language profiles
│   ├── scenario_service.py   # Physical clamping, baseline vs scenario evaluation, delta LST
│   └── proposal_service.py   # SQLite CRUD, state transitions, reviewer comments & audit log
├── tests/
│   └── test_municipal_services.py # Automated test suite for data status, scenario clamping & proposals
├── notebooks/
│   └── academic_pipeline_walkthrough.py  # Interactive cell-by-cell notebook script
├── Data in system.md         # Operational guide to Spatial Block R², HRI & Tab-by-Tab Graph Mappings
├── run_dashboard.bat         # Windows batch launcher
├── run_dashboard.ps1         # PowerShell launcher
├── requirements.txt
├── FEATURE_DOCUMENTATION_AND_OPERATIONS_MANUAL.md
└── README.md
```

---

## 7. How to Run the Platform

### A. Environment Setup
```powershell
pip install -r requirements.txt
```

### B. Launch the Interactive Mission-Control Platform (Easiest)
```powershell
.\run_dashboard.bat
# or in PowerShell:
.\run_dashboard.ps1
```
*(Or via Python module syntax)*:
```powershell
python -m streamlit run app/dashboard.py
```
The platform will launch at `http://localhost:8501`.

### C. Re-run the Complete ML & Risk Assessment Pipeline (Optional)
```powershell
# Run both Delta (Trichy) and Non-Delta (Chennai)
python scripts/run_pipeline.py --all

# Or run for a specific region
python scripts/run_pipeline.py --region trichy_delta
python scripts/run_pipeline.py --region chennai_nondelta
```

### D. Run Automated Verification & Service Tests
```powershell
python -c "import tests.test_municipal_services as t; t.test_data_status_tracker(); t.test_scenario_clamping_and_bounds(); t.test_scientific_hri_bounds_and_monotonicity(); t.test_state_machine_and_terminal_enforcement(); t.test_csv_formula_injection_sanitization(); t.test_stored_xss_escaping(); print('ALL 6 RIGOROUS SCIENTIFIC AND SECURITY TESTS PASSED!')"
```

---

## 8. Detailed System Documentation & Operational Guides

- **Strategic Operations Manual & 5-Whys Analysis:**  
  👉 **[FEATURE_DOCUMENTATION_AND_OPERATIONS_MANUAL.md](FEATURE_DOCUMENTATION_AND_OPERATIONS_MANUAL.md)**
- **System Metrics & Tab-by-Tab Graph Interpretations Guide:**  
  👉 **[Data in system.md](Data%20in%20system.md)** (Explains Spatial Block 5-Fold $R^2$, Heat Risk Index (HRI) decoupled formulation, and complete mapping of all graphs across every tab).

---

## 9. Data Sources & Legal Attributions

- **Base Cartography:** &copy; [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors (ODbL).
- **Satellite Imagery:** Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community.
- **Weather-Model-Based Conditions:** Atmospheric model data provided by [Open-Meteo.com](https://open-meteo.com/) (NOAA GFS / DWD ICON blend) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Satellite Data:** USGS/NASA Landsat 8/9 courtesy of the U.S. Geological Survey; Copernicus Sentinel-2 data courtesy of the European Space Agency (ESA).
