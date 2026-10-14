# MUNICIPAL HEAT WATCH: Comprehensive Feature Documentation & Strategic Operations Manual

> **Academic & Applied Research Prototype Notice:**  
> This system is an applied research decision-support prototype developed for academic and methodological demonstration. The workflows and screens reflect intended operational processes for target users (urban planners, civic engineers, and disaster analysts), but have **not yet been formally validated, endorsed, or adopted by the Municipal Corporations of Tiruchirappalli, Greater Chennai Corporation, or the Government of Tamil Nadu**.

---

## 1. The Strategic Foundation: What, Why, How, and For Whom

### WHAT Are We Doing?
We are building **Municipal Heat Watch**: a spatial decision-support and workflow prototype. It couples satellite Earth Observation (Landsat 8/9, Sentinel-2), numerical atmospheric reanalysis (ERA5), weather-model-based atmospheric conditions (Open-Meteo GFS/ICON blend), and socio-demographic vulnerability data (WorldPop) to produce an actionable, ward-level **Heat Action Plan (HAP)** exploratory command platform for two distinct urban contexts in Tamil Nadu.

### WHY Are We Doing This?
Urban heat anomalies present severe public health hazards in rapidly urbanizing Indian cities. However, traditional academic projects often stop at producing static heat maps or global regression metrics ($R^2$, RMSE) that are difficult for civic planners to translate into operational priorities. **We are building this to bridge the gap between spatial machine learning and municipal administration**, translating spatial predictions into prioritized, ward-specific civic proposals with physical constraints and accountability tracking.

### HOW Are We Doing This?
1. **Decoupled Scientific Risk Modeling:** We model physical hazard exposure ($E$, derived from satellite LST and urban form) separately from human vulnerability ($V$, sensitive age groups, poverty, healthcare distance), combining them via the standard composite formula:
   $$\text{Heat Risk Index (HRI)} = \sqrt{\text{Exposure Score} \times \text{Vulnerability Score}}$$
2. **Spatial Block Cross-Validation:** We evaluate Random Forest, XGBoost, and Ridge models across 5 geographic spatial blocks to minimize spatial autocorrelation data leakage.
3. **Plain-Language Explainable AI (XAI):** We translate SHAP (Shapley Additive Explanations) mathematical attributions into non-technical factor breakdowns relative to the city baseline (e.g. *"+2.8°C model contribution associated with high built-up density"*).
4. **Physically Clamped Scenario Simulation:** We provide a what-if simulator that strictly clamps inputs within the model's empirical training domain, ensuring simulated cooling benefits ($\Delta \text{LST}$) remain within observed physical limits.
5. **Persistent Relational Database:** We back Heat Action Plan interventions with an embedded SQLite database (`data/app_database.db`), tracking proposals across a defined administrative lifecycle (`Draft` &rarr; `Submitted` &rarr; `Approved` &rarr; `Completed`) with persistent local database action logs.

### FOR WHOM Are We Doing This?
*(Intended user personas for prototype evaluation; not yet formally validated by civic authorities)*:
1. **Urban Local Body Administrators & Disaster Management Officers:** Decision-makers who coordinate heat emergency advisories and resource allocation during summer heatwaves.
2. **Urban Primary Health Centre (UPHC) Officers:** Medical officers who monitor vulnerable demographics to pre-position oral rehydration salts (ORS) and cooling beds.
3. **Public Works & Urban Forestry Planners:** Civic engineers who design cool-roof retrofits, urban tree planting (Miyawaki pockets), and transit shade structures.
4. **Vulnerable Communities (Indirect Beneficiaries):** Daily-wage laborers, street vendors, the elderly, and children in dense informal settlements who are disproportionately exposed to thermal stress.

---

## 2. The 5-Whys Incremental Root-Cause Analysis

To achieve absolute conceptual clarity, we examine the problem through the **5-Whys framework**:

### Why 1: Why are we building this system?
> **Because extreme urban heat causes preventable morbidity and mortality, yet city administrations lack neighborhood-level tools to prioritize localized interventions.**
*City-wide weather forecasts provide a single uniform temperature reading, masking the reality that dense informal settlements can experience significantly higher surface temperatures than shaded riverbanks.*

### Why 2: Why does predicting surface temperature (LST) not solve heatwave mortality on its own?
> **Because physical temperature alone does not define an emergency; human vulnerability does.**
*An uninhabited industrial corridor with an LST of 44°C requires a different municipal response than a residential ward with an LST of 41°C housing 15,000 low-income residents, 20% senior citizens, and limited healthcare access. Without decoupling exposure from vulnerability, civic resources cannot be targeted effectively.*

### Why 3: Why can municipal officials not act on standard machine learning models?
> **Because standard ML models output abstract statistics (coefficients, SHAP waterfall plots, $R^2$) that non-technical administrators cannot interpret or defend in policy notes.**
*A municipal planner cannot justify an emergency expenditure based on raw feature weights. They require plain-language factor attributions: "Ward 32 exhibits high heat exposure primarily due to dense impervious surfaces (57% footprint) combined with severe canopy deficit (NDVI 0.04)."*

### Why 4: Why must interventions be tied to a persistent governance workflow rather than simulated dashboard alerts?
> **Because monitoring dashboards are purely observational, whereas civic administration requires operational accountability.**
*If an alert disappears when a browser tab closes, nobody is accountable. In municipal governance, an alert must become a tracked proposal with an assigned department (e.g., Public Works), a budget estimate, an urgency level, a formal approval status (`Submitted` &rarr; `Approved`), and a persistent local database action log.*

### Why 5: Why is this critical specifically for Tamil Nadu's dual context (Delta vs Coastal Metro)?
> **Because the Cauvery Delta (Tiruchirappalli) and the Coastal Metropolis (Chennai) represent fundamentally different urban heat physics that require distinct local strategies.**
*Tiruchirappalli is an inland riverine gateway where drying agricultural wetlands and river sandbanks create dry summer daytime thermal anomalies (+44.6°C LST). In contrast, Chennai is a dense coastal corridor where maritime humidity creates oppressive apparent Heat Indices despite maritime moderation. Modeling both environments demonstrates how spatial ML must adapt to localized geographic drivers.*

---

## 3. Spatial Extent & Administrative Ward Clarification

| Parameter | Tiruchirappalli Study Area | Chennai Study Area |
| :--- | :--- | :--- |
| **Study Classification** | Delta Urban-Agricultural Fringe | Non-Delta Coastal Urban Heat Corridor |
| **Grid Cell Count** | 5,040 cells (250m × 250m) | 4,636 cells (250m × 250m) |
| **Total Spatial Extent** | **~315.0 km²** (Bounding Box: $78.62^\circ\text{E} - 78.78^\circ\text{E}$, $10.72^\circ\text{N} - 10.88^\circ\text{N}$) | **~289.75 km²** (Bounding Box: $80.18^\circ\text{E} - 80.32^\circ\text{E}$, $12.98^\circ\text{N} - 13.15^\circ\text{N}$) |
| **Wards Represented** | **65 Administrative Wards** | **100 Sampled Wards** |
| **Jurisdictional Note** | Encompasses the statutory Tiruchirappalli City Corporation (~167 km²) plus peri-urban agricultural delta buffer zones. | Represents a 100-ward core urban heat island study corridor sampled from the Greater Chennai Corporation's 200 statutory wards (426 km²). |

---

## 4. Detailed Feature-by-Feature Operational Manual

### Global Top Brand Header & Provenance Status
- **Location:** Top row of every screen.
- **Components:**
  - Platform branding with weather-model feed and demonstration mode badges.
  - **City Corporation Switcher:** Toggles between Tiruchirappalli and Chennai. Updates coordinate extents, loads regional models, and recalculates ward rankings.
  - **Weather-Model-Based Conditions:** Ingests atmospheric model conditions (2m dry-bulb temperature, relative humidity, wind speed, and apparent heat index) from Open-Meteo (NOAA GFS / DWD ICON blend) displaying valid time and retrieval time.
  - **1-Click CSV Export:** Downloads the active city's ward assessment dataset with formula injection sanitization applied.

---

### Tab 1: Executive Summary & Administrative Ranking
- **Components:**
  1. **5 KPI Metric Deck:** Surveyed Wards, Mean/Peak Surface Temperature, Emergency Priority Wards, Spatial Model $R^2$ (Cross-Validated), and Aggregated Exposed Population (avoiding grid double-counting).
  2. **Administrative Ward Risk Rankings Table:** Searchable rankings table (Rank #1 = highest risk) displaying Ward ID, Name, Risk Tier pill, composite HRI, mean LST, and population density. Rendered with clean unindented HTML to prevent Markdown code block artifacts.
  3. **Active Ward Quick Assessment Card:** Synchronized card showing the selected ward's vital statistics and top non-causal statistical drivers.

---

### Tab 2: Ward Risk Profile & Assessment (The Core Screen)
- **Structure (5 Dedicated Sections):**
  1. **Section 1 (Situation):** Risk Category, Priority Rank, Satellite LST, Decoupled HRI score, and data confidence rating.
  2. **Section 2 (People & Socio-Demographic Vulnerability):** Estimated residents, sensitive age groups (children $<5$ and elderly $>65$), low-income household ratio, and healthcare distance.
  3. **Section 3 (Why This Ward? - Non-Causal XAI):** Explains model-associated factors relative to the city baseline:
     - *Heating Drivers:* Identifies statistical associations with high built-up density and canopy deficit.
     - *Cooling Buffers:* Identifies statistical associations with vegetation and river/canal proximity.
     - *Caveat:* Explicitly notes that SHAP measures model attribution, not physical causality.
  4. **Section 4 (Recommended Heat Action Plan Measures):** Prioritized protocols aligned with risk tiers (e.g. hydration kiosks, cool roof coatings, outdoor labor rest advisories per NDMA 2019 guidelines).
  5. **Section 5 (Next Step CTA: Formalize Proposal):** Form that pre-populates ward context and submits a prototype proposal into the SQLite database.

---

### Tab 3: Spatial Heat & Risk Map
- **Components:**
  1. **Layer Display:** Toggle between Ward Heat Risk Index (WHRI), Mean LST (°C), and Vulnerability Score.
  2. **Free Basemap Switcher:** Toggle between **OpenStreetMap** (`© OpenStreetMap contributors`) and **Esri World Imagery** (`Tiles © Esri`). All tile provider attributions are visible and compliant with licensing terms.
  3. **Interactive GeoJSON Layer:** Color-coded choropleths showing all municipal wards with interactive tooltips. Active ward is highlighted with dynamic styling.

---

### Tab 4: Scenario Workspace (What-If Intervention Simulator)
- **Components:**
  1. **Pre-Packaged Municipal Interventions:** `Cool Roofs`, `Miyawaki Urban Forestry`, `Permeable De-paving`, `Canal Restoration`, and `Custom`.
  2. **Domain Boundary Clamping:** Strictly enforces physical bounds defined in `src/config.py` (e.g. Albedo clamped to $[0.05, 0.55]$, NDVI to $[-0.20, 0.85]$) to prevent unscientific extrapolations.
  3. **Model Evaluation:** Compares baseline vs scenario to predict $\Delta \text{LST}$ (°C cooling) and updated Heat Risk Index. Logs physical model assumptions and limitations.

---

### Tab 5: Explainable AI (XAI & SHAP)
- **Components:**
  1. **Non-Technical Attribution (Default View):** Plain-language factor impact cards categorizing variables into heating drivers vs cooling mitigators relative to baseline.
  2. **Expandable Technical Accordion:** Live local SHAP Waterfall plot (<10ms), global feature importance bar plot, beeswarm distribution, and non-linear dependence tipping points ($\text{NDVI} > 0.32$).

---

### Tab 6: HAP Proposals & Review Tracker (Persistent SQLite Database)
- **Components:**
  1. **Persistent SQLite Database (`src/proposal_service.py`):** Stores records across application restarts in `data/app_database.db`.
  2. **Server-Side State Machine:** Enforces valid transitions (`Draft` $\to$ `Submitted` $\to$ `Approved` $\to$ `Completed`). `Completed` proposals are terminal and cannot be reopened.
  3. **Reviewer Action Panel:** Allows simulated review and status updates. Includes a clear demonstration mode disclaimer noting role selection is unauthenticated.
  4. **Security Protections:**
     - All user-submitted text rendered via `html.escape()` to prevent Stored XSS.
     - All SQL queries use parameterized `?` placeholders.
     - CSV export applies `sanitize_for_csv()` to neutralize Spreadsheet Formula Injection (`=`, `+`, `-`, `@`).
  5. **Chronological Audit Trail:** Persistent local database action log of timestamps, reviewer roles, status transitions, and remarks.

---

### Tab 7: Data Freshness, Lineage & Provenance Tracker
- **Components:**
  1. **Demonstration Mode Notice:** Clarifies that spatial grids are calibrated demonstration datasets, while weather-model-based conditions are retrieved via an API forecast feed.
  2. **Sensor Lineage Audit Cards:** Details source, observation date, spatial resolution, and latency for Landsat 8/9, Sentinel-2, ERA5, WorldPop, OSM, and Open-Meteo.
  3. **Observation Timeline:** Compares peak heatwave baseline scenes against early summer and historical archive scenes.

---

### Tab 8: Trichy vs Chennai (Delta vs Non-Delta Comparative Analysis)
- **Components:**
  1. **Comparative Matrix Table:** Evaluates Geographic Context, Natural Buffers (Cauvery River vs Sea Breeze), Urban Encroachment patterns, and Peak Hotspot characteristics.
  2. **Distribution Histograms:** Side-by-side frequency distributions of Land Surface Temperature across both cities.

---

### Tab 9: Live Weather API & Architecture Guide
- **Components:**
  1. **Live Weather-Model Conditions Ingestion:** Fetches atmospheric model conditions from Open-Meteo (NOAA GFS / DWD ICON blend) displaying valid time and retrieval time.
  2. **Scientific Scope Notice:** Clearly clarifies:
     > *[ILLUSTRATION ONLY]: In this version, weather-model-based conditions are monitored strictly as real-time ambient context and are NOT applied to modify pre-computed static satellite LST or ward risk scores. Feeding single-point live weather into static satellite models without multi-temporal recalibration would be scientifically invalid.*
  3. **Developer Reference:** Clean Python architecture demonstrating conceptual multi-temporal ingestion.
  4. **Methodological Roadmap:** Highlights planned extensions (Leaflet-Draw custom polygon analysis, automated PDF briefings). Removed unrelated grid AC demand scope creep.

---

## 5. Verification & Testing Runbook

### Automated Test Suite Execution
Execute the automated test suite covering scientific bounds, state machine rules, and security guards:
```powershell
python -c "import tests.test_municipal_services as t; t.test_data_status_tracker(); t.test_scenario_clamping_and_bounds(); t.test_scientific_hri_bounds_and_monotonicity(); t.test_state_machine_and_terminal_enforcement(); t.test_csv_formula_injection_sanitization(); t.test_stored_xss_escaping(); print('ALL 6 RIGOROUS SCIENTIFIC AND SECURITY TESTS PASSED!')"
```

### Verified Test Assertions:
1. **Provenance & Timeline:** Confirmed layers are audited and timeline scenes load.
2. **Physical Clamping:** Confirmed features outside empirical bounds are safely clamped.
3. **Scientific HRI Bounds & Monotonicity:** Confirmed $0.0 \le \text{HRI} \le 1.0$ across all valid inputs, and decreasing exposure strictly decreases HRI.
4. **State Machine Terminal Enforcement:** Confirmed legal transitions (`Draft` $\to$ `Submitted` $\to$ `Approved` $\to$ `Completed`) succeed, while attempting to reopen a `Completed` proposal raises a server-side `ValueError`.
5. **CSV Formula Injection:** Confirmed dangerous prefixes (`=`, `+`, `-`, `@`) are escaped with an apostrophe.
6. **Stored XSS Escaping:** Confirmed malicious script tags are escaped to harmless entities (`&lt;script&gt;`).
