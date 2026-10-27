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

### Global Navigation & Header Architecture
- **Sideways Navigation Sidebar:**
  - Optimized vertical layout located in the sidebar for intuitive one-hand navigation.
  - Streamlined spacing with clean uppercase section header (`NAVIGATION MENU`).
  - Elimination of duplicate widget labels using dedicated CSS selector isolation (`div[data-testid="stRadio"] > label { display: none !important; }`).
- **Global Top Brand Header & Operational Status:**
  - **Platform Branding:** "Municipal Heat Watch - Spatial Screening System & Decision Support".
  - **City Corporation Switcher:** Instant toggle between Tiruchirappalli (Cauvery Delta) and Chennai (Coastal Metro), updating spatial extents, regional ML inference models, and ward risk rankings dynamically.
  - **Model Operational Badge:** Active surveillance indicator showing real-time ML inference readiness. (Extraneous static demo badges such as Level 2 Heat Screening have been removed for operational clarity).
  - **1-Click CSV Export:** Generates an immediate download of the active city's ward risk assessment dataset with formula injection sanitization applied.

---

### Tab 1: Dashboard (`📊 Dashboard`)
- **Core Purpose:** High-level operational command center for municipal executives, disaster coordinators, and city engineers.
- **Components & Layout:**
  1. **5 KPI Telemetry Deck:**
     - *Total Surveyed Wards:* Statutory and peri-urban wards analyzed (65 in Tiruchirappalli, 100 in Chennai).
     - *Mean & Peak Surface Temperature:* Satellite-observed radiometric surface temperatures (°C).
     - *Priority Action Wards:* Wards exceeding the municipal emergency threshold ($\text{WHRI} > 0.50$).
     - *Spatial Block 5-Fold Model $R^2$:* Cross-validated spatial predictive accuracy ($R^2 = 0.876$ for Trichy, $R^2 = 0.950$ for Chennai).
     - *Exposed Population:* Total population residing in priority heat hazard zones, formatted cleanly using `k` up to 900k (e.g. `850k`) and transitioning seamlessly to `M` above 900k (e.g. `1.25M`) without double-counting.
  2. **Administrative Ward Rankings Table:**
     - Scrollable, fixed-height 8-row viewport table displaying Ward ID, Name, Priority Tier, composite HRI, Mean LST (°C), and population density (/km²).
     - Instant real-time text filter and search bar for quick ward retrieval.
  3. **Synchronized Active Ward Selector:**
     - Positioned immediately above the active ward assessment header (e.g., right above "Tiruch Ward 23") adjacent to the table, creating a seamless visual hierarchy between selection and deep-dive analytics.
  4. **Active Ward Quick Assessment Panel:**
     - Displays the selected ward's vital statistics (Rank, Priority Tier, WHRI score, Population at Risk, Mean LST).
  5. **Ward-Specific Live Weather Feed (Open-Meteo Integration):**
     - Automatically resolves the exact geographic centroid coordinates of the selected ward and queries the Open-Meteo atmospheric model (NOAA GFS / DWD ICON blend) in real time.
     - Displays:
       - **Dry-Bulb Air Temperature (°C):** Current ambient 2m air temperature.
       - **Relative Humidity (%):** Atmospheric moisture content.
       - **Apparent Heat Index (°C):** Physiological "feels-like" temperature combining heat and humidity.
       - **Surface Wind Speed (km/h):** Natural convective ventilation.
     - Consolidates ambient live weather directly into the dashboard, eliminating the need for a separate live tab.

---

### Tab 2: Ward Risk Profile & Assessment (`📋 Ward Risk Profile`)
- **Core Purpose:** Deep-dive diagnostic profile enabling urban planners to understand *why* a specific ward is hot and *who* is affected.
- **Dynamic Graphical Visualizations:**
  1. **Thermal Radiation Context (Landsat-9):**
     - Horizontal bar chart comparing Peak Pixel LST, Ward Mean LST, and Regional City Baseline (°C) to identify localized micro-hotspots.
  2. **Physical Environmental Covariates:**
     - Horizontal bar chart displaying Distance to Water (km), Built-Up Index (NDBI x100), Tree Canopy (NDVI x100), and Ground Building Density (%).
  3. **People & Socio-Demographic Vulnerability:**
     - *Demographic Cohorts Graph:* Dynamic horizontal bar chart breaking down the resident population into general cohort, seniors (>65), and young children (<5) with headcounts and percentages.
     - *Socio-Demographic Vulnerability Factors Graph:* Dynamic horizontal bar chart depicting Composite Vulnerability weight, Child Ratio, Senior Ratio, and Low-Income Household percentage.
     - *Live Ward Reactivity:* All graphs dynamically update and re-render in real time whenever the user switches the active ward.
  4. **Why This Ward? (Key Thermal & Vulnerability Drivers):**
     - Structured physical attribution cards categorizing local factors into heating drivers vs. cooling buffers.
  5. **Recommended Heat Action Plan (HAP) Measures:**
     - Elevated, high-contrast, eye-catching action container highlighting priority municipal protocols:
       - Cool roof reflective coatings for low-income informal housing.
       - Ward-level drinking water kiosks and Oral Rehydration Salts (ORS) distribution centers.
       - Outdoor labor shade regulations and afternoon work shifts during peak solar hours.
       - Emergency cool shelter activations in community halls and primary schools.
  6. **Formalize Proposal Workflow:**
     - Pre-populates ward context into an administrative form to submit actionable proposals directly into the persistent SQLite database.

---

### Tab 3: Spatial Heat & Risk Map (`🗺️ Spatial Heat Map`)
- **Core Purpose:** Interactive geospatial mapping for territorial spatial screening and zone-wide resource allocation.
- **Components:**
  1. **Interactive GeoJSON Choropleth Map:**
     - High-performance Leaflet choropleth map color-coding all administrative wards into 5 priority tiers:
       - 🔴 **Extreme Priority:** $\text{HRI} \ge 0.70$ (Red)
       - 🟠 **Very High:** $0.55 \le \text{HRI} < 0.70$ (Orange)
       - 🟡 **High:** $0.40 \le \text{HRI} < 0.55$ (Amber)
       - 🟢 **Moderate:** $0.25 \le \text{HRI} < 0.40$ (Green)
       - 🔵 **Low:** $\text{HRI} < 0.25$ (Blue)
     - Interactive hover tooltips displaying Ward ID, Ward Name, Priority Rank, Risk Tier, Mean LST (°C), and WHRI score.
  2. **Basemap Style Switcher:**
     - Seamless toggle between **OpenStreetMap** (Clean, Cartographic) and **Esri World Imagery** (High-Resolution Satellite Photography) without requiring third-party API keys or leaving map watermarks.

---

### Tab 4: Scenario Workspace (`🔮 Scenario Simulation`)
- **Core Purpose:** Predictive sandbox allowing municipal planners and private developers to simulate cooling interventions and evaluate the microclimatic impact of new construction.
- **Features Two Interactive Simulation Modes:**

#### Mode A: Ward Cooling Interventions (HAP)
- *Intuitive Real-World User Inputs:* Replaces abstract mathematical multipliers with tangible civic units:
  - **High-Albedo Cool Roof Coating:** Slider for Number of Houses/Buildings Coated (100 to 5,000 houses) calculating treated roof area in m² and solar reflectance uplift.
  - **Miyawaki Forest & Tree Canopy:** Slider for Native Trees/Saplings Planted (250 to 10,000 trees) calculating added canopy hectares and evapotranspirative cooling.
  - **Permeable De-paving:** Slider for Impervious Concrete/Asphalt De-paved (5,000 to 100,000 m²) calculating soil moisture recovery.
  - **Water Corridor Restoration:** Slider for Canal/Wetland Buffer Restored (200 to 4,000 linear meters) calculating riparian thermal buffer expansion.
  - **Comprehensive HAP:** Municipal Resilience Coverage slider (10% to 100% ward footprint).
- *Dual-Bar Evaluation Graph:*
  - Side-by-side Matplotlib chart comparing **Baseline vs. Scenario** for Surface LST (°C) and Heat Risk Index (HRI).
  - Dynamic delta callout badges showing cooling impact ($\Delta \text{LST}$ in °C and $\Delta \text{HRI}$).
- *Model Domain Check Callout:*
  - Real-time sanity check confirming whether simulated inputs remain within observed empirical satellite boundaries or if boundary clamping was applied.

#### Mode B: New Building Construction Impact Simulator (Brand New Feature)
- *Microclimate & Environmental Impact Analysis:* Models how a proposed residential, commercial, or industrial building will alter the local thermal environment and urban hydrology.
- *Site & Architectural Inputs:*
  - **Ground Plot Footprint (sq. ft):** Slider from 2,000 to 100,000 sq. ft representing land coverage.
  - **Number of Floors (Stories):** Slider from 1 to 30 stories calculating total built-up floor area.
  - **Approximate Residents / Occupants:** Slider from 10 to 2,000 people calculating human density and internal heat load.
  - **Parking Location:** Radio toggle between *Inside Building / Underground* (no asphalt surface penalty) vs. *Surface Asphalt Lot* (+1.2°C surface heat penalty, +25% runoff penalty).
  - **Basement Excavation:** Radio toggle between *Yes* (minor thermal mass buffering) vs. *No*.
- *Green Building Mitigation Toggles:*
  - High-Albedo Cool Roof Coating (SRI $\ge$ 82, -1.8°C roof cooling).
  - Permeable Interlocking Pavers on Driveways (-35% stormwater runoff).
  - 20% Native Tree Buffer & Canopy Setback (-0.9°C microclimate buffer).
  - High-Efficiency HVAC with Heat Recovery / Thermal Storage (-40% waste heat rejection).
- *Dual-Panel Comparison Charts:*
  - **Chart 1 (Local Surface Heat Impact):** Three-bar comparison displaying **Ward Baseline LST**, **Standard Unmitigated Construction**, and **Green-Mitigated Construction**.
  - **Chart 2 (Environmental Stress Index):** Three-dimensional comparison of **Stormwater Runoff Volume**, **Impervious Land Fraction**, and **HVAC Waste Heat Rejection** under unmitigated vs. mitigated designs.
- *Environmental Analysis & Municipal Measures:*
  - Plain-language environmental brief estimating net temperature delta and stormwater impact.
  - Actionable municipal planning by-law conditions (e.g. mandatory cool-roof coating on flat roofs, mandatory rainwater harvesting percolation pits, minimum 15% green setback buffer).

---

### Tab 5: Explainable AI (`🧠 Explainable AI (SHAP)`)
- **Core Purpose:** Demystifies the machine learning model for non-technical stakeholders, municipal leaders, and common citizens.
- **Key Features:**
  1. **Common-Man Readable Variable Names:**
     - Translates raw technical feature names into clear, intuitive terminology:
       - `Building_Density_pct` $\to$ **Building Footprint (% Ground Covered)**
       - `NDVI` $\to$ **Green Vegetation & Tree Canopy (NDVI)**
       - `NDBI` $\to$ **Built-Up Concrete & Asphalt Index (NDBI)**
       - `Albedo` $\to$ **Surface Sun Reflectance (Albedo)**
       - `Dist_to_Water_m` $\to$ **Distance to Water Channel (meters)**
       - `Elevation_m` $\to$ **Ground Elevation (meters)**
       - `Road_Density` $\to$ **Road & Pavement Network Density**
  2. **Step-by-Step Graph Interpretation Guide:**
     - Clear instructional callout box explaining:
       - **Baseline Starting Point ($E[f(x)]$):** The city-wide average baseline temperature across all surveyed wards.
       - **Red Bars (+):** Local factors that heat the ward up above city baseline (e.g. dense concrete, high building footprint, low vegetation).
       - **Blue Bars (-):** Local factors that cool the ward down below city baseline (e.g. nearby river channels, high tree canopy, high reflective surfaces).
       - **Final Result ($f(x)$):** The model's final predicted surface temperature for that specific ward.
  3. **Local SHAP Waterfall Plot:**
     - Sub-10ms TreeExplainer computation displaying exact degree contributions for the active ward.
  4. **Dynamic Ward Takeaways:**
     - Plain-language summary card identifying the primary local heating drivers and recommended municipal cooling levers.
  5. **Global Model Insights:**
     - Global Mean Feature Importance bar chart and Beeswarm distribution plot with non-technical captions explaining city-wide microclimate tipping points.

---

### Tab 6: HAP Proposals & Review Tracker (`📝 HAP Action Tracker`)
- **Core Purpose:** Closes the loop between spatial AI analysis and administrative governance.
- **Components:**
  1. **Persistent SQLite Database (`src/proposal_service.py`):**
     - Stores civic proposals, budgets, target departments, and review statuses across server restarts in `data/app_database.db`.
  2. **Server-Side State Machine:**
     - Enforces rigorous administrative lifecycle transitions:  
       `Draft` $\to$ `Submitted` $\to$ `Under Review` $\to$ `Approved` $\to$ `Completed` / `Rejected`.
     - Terminal state enforcement prevents unauthorized tampering with completed proposals.
  3. **Reviewer Action Panel:**
     - Administrative interface enabling municipal officers to review proposals, approve budgets, and append official remarks.
  4. **Security & Data Sanitization:**
     - Built-in Stored XSS escaping (HTML entity encoding) and CSV Formula Injection sanitization (escaping `=`, `+`, `-`, `@`).
  5. **Chronological Audit Trail:**
     - Persistent audit log recording timestamps, reviewer roles, remarks, and state transitions.

---

### Tab 7: Data Freshness & Provenance (`🔄 Data Provenance`)
- **Core Purpose:** Transparency and scientific traceability for researchers, auditors, and municipal oversight committees.
- **Components:**
  1. **Multi-Source Lineage Cards:**
     - Full documentation of satellite platforms (Landsat-9, Sentinel-2), numerical models (ERA5, Open-Meteo), and socio-demographic databases (WorldPop 2024).
  2. **Audit Verification:**
     - Calibration formulas, spatial resolutions, cloud-cover filtering, and update latencies.
  3. **Operational Scope Clarification:**
     - Notes that real-time Open-Meteo weather feeds provide ambient situational context and are physically decoupled from the static spatial satellite regression models.

---

## 5. Companion Documentation & Knowledge Base

For comprehensive mathematical foundations, metric definitions, and visual mapping guides, refer to:
- **[Data in system.md](file:///c:/Users/sathi/.antigravity-ide/Projects/Data%20in%20system.md)** (or [Data in system](file:///c:/Users/sathi/.antigravity-ide/Projects/Data%20in%20system)):
  - **Section 1: Spatial Block 5-Fold Cross-Validation $R^2$:** Explains geographic block partitioning to eliminate spatial autocorrelation data leakage.
  - **Section 2: Heat Risk Index (HRI):** Explains geometric decoupling ($\text{HRI} = \sqrt{E \times V}$), vulnerability weighting, and why additive sums fail in civic operations.
  - **Section 3: System Data Summary Matrix:** Details all 7 primary data layers, sources, resolutions, and update frequencies.
  - **Section 4: Tab-by-Tab Graph Interpretations & Mapping Guide:** Complete, structured breakdown mapping all 11+ graphs across Tabs 1 through 7, explaining their variables, how to read them, and civic action takeaways.

---

## 6. Verification & Testing Runbook

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
