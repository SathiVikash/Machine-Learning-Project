"""
Configuration module for Urban Heat Vulnerability and Explainable ML Assessment.
Defines study areas (Cauvery Delta vs Non-Delta), spatial parameters, feature sets,
and administrative governance thresholds.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"

for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR, MODELS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Study Area Profiles: Delta vs Non-Delta
REGIONS = {
    "trichy_delta": {
        "name": "Tiruchirappalli (Cauvery Delta Gateway)",
        "type": "Delta Urban Fringe Study Area",
        "description": "Metropolitan research extent (5,040 cells, ~315 km²) encompassing the statutory corporation (~167 km²) and peri-urban agricultural delta fringe.",
        "center_lat": 10.7905,
        "center_lon": 78.7047,
        "utm_crs": "EPSG:32644",
        "bounds": {
            "min_lon": 78.62,
            "max_lon": 78.78,
            "min_lat": 10.72,
            "max_lat": 10.88,
        },
        "default_grid_size_m": 250,
        "wards_count": 65,
    },
    "chennai_nondelta": {
        "name": "Chennai Metropolitan Area",
        "type": "Non-Delta Coastal Heat Corridor",
        "description": "Designated 100-ward urban heat island study corridor (~290 km²) sampled from Greater Chennai Corporation's 200 statutory wards.",
        "center_lat": 13.0827,
        "center_lon": 80.2707,
        "utm_crs": "EPSG:32644",
        "bounds": {
            "min_lon": 80.18,
            "max_lon": 80.32,
            "min_lat": 12.98,
            "max_lat": 13.15,
        },
        "default_grid_size_m": 250,
        "wards_count": 100,
    }
}

# Features used for ML Modeling of Land Surface Temperature (Heat Exposure)
SPECTRAL_FEATURES = ["NDVI", "NDBI", "NDWI", "Albedo"]
TERRAIN_FEATURES = ["Elevation_m", "Slope_deg"]
URBAN_FORM_FEATURES = [
    "Road_Density_km_km2",
    "Building_Density_pct",
    "Dist_to_Water_m",
    "Dist_to_Park_m"
]
CLIMATE_AUX_FEATURES = ["Air_Temp_ERA5", "Relative_Humidity_pct"]

EXPOSURE_FEATURES = (
    SPECTRAL_FEATURES +
    TERRAIN_FEATURES +
    URBAN_FORM_FEATURES +
    CLIMATE_AUX_FEATURES
)

TARGET_VARIABLE = "LST_Celsius"

# Vulnerability Indicators (WorldPop + Municipal Socio-Demographics)
VULNERABILITY_FEATURES = [
    "Population_Density_per_km2",
    "Elderly_Ratio_pct",       # Age > 65
    "Children_Ratio_pct",      # Age < 5
    "Low_Income_Household_pct",# Socioeconomic vulnerability
    "Distance_to_Hospital_m"   # Healthcare access deficit
]

# Risk Classifications
RISK_CATEGORIES = ["Low", "Moderate", "High", "Very High", "Extreme Priority"]
RISK_COLORS = {
    "Low": "#2b83ba",
    "Moderate": "#abdda4",
    "High": "#fdae61",
    "Very High": "#ea5545",
    "Extreme Priority": "#7d0025"
}

# Administrative Decision Matrix: Policy Actions per Risk Category
ADMIN_POLICY_ACTIONS = {
    "Extreme Priority": [
        "Deploy temporary shaded public cooling shelters and ORS hydration kiosks.",
        "Issue advisory recommending suspension of heavy outdoor manual labor during peak solar hours (12:00 - 15:30) per NDMA Guidelines (2019).",
        "Propose cool-roof reflective coating subsidies for vulnerable informal housing clusters.",
        "Deploy municipal water-misting tankers along primary road heat corridors."
    ],
    "Very High": [
        "Priority allocation of municipal urban forestry and Miyawaki green pocket funding.",
        "Retrofit public bus shelters with reflective roofing and shaded vegetation.",
        "Targeted community health volunteer visits to monitor senior citizens and children."
    ],
    "High": [
        "Enforce green building codes and minimum 15% permeable surface mandates on new approvals.",
        "Implement rainwater percolation and neighborhood water body restoration to enhance evaporative cooling.",
        "Public heatwave advisory banners and SMS notifications to ward residents."
    ],
    "Moderate": [
        "Regular monitoring of vegetation health (NDVI) via seasonal satellite surveillance.",
        "Maintain existing civic parks and avenue tree canopies."
    ],
    "Low": [
        "Designate as climate-resilient ecological buffer zones (riverbanks, protected green zones).",
        "Prevent uncurbed encroachment and conversion of agricultural/wetland areas."
    ]
}

# ==============================================================================
# DATASET PROVENANCE & FRESHNESS SPECIFICATIONS
# ==============================================================================
DATASET_PROVENANCE = {
    "lst_satellite": {
        "name": "Land Surface Temperature (LST)",
        "source": "USGS / NASA Landsat 8/9 TIRS (Thermal Infrared Sensor)",
        "spatial_resolution": "30m (interpolated to 250m grid)",
        "cadence": "16-day revisit",
        "nominal_latency_days": 2,
        "description": "Radiometrically calibrated top-of-atmosphere brightness temperature converted via Split-Window / Radiative Transfer Equation.",
        "type": "Remote Sensing (Satellite Observation)"
    },
    "surface_indices": {
        "name": "Multispectral Indices (NDVI, NDBI, NDWI, Albedo)",
        "source": "ESA Copernicus Sentinel-2 MSI (Level-2A BOA)",
        "spatial_resolution": "10m - 20m",
        "cadence": "5-day revisit",
        "nominal_latency_days": 1,
        "description": "Atmospherically corrected bottom-of-atmosphere surface reflectance with cloud-masking (<10% cloud coverage).",
        "type": "Remote Sensing (Satellite Observation)"
    },
    "climate_reanalysis": {
        "name": "Atmospheric Boundary (Air Temp, Relative Humidity)",
        "source": "ECMWF ERA5-Land Reanalysis Hourly",
        "spatial_resolution": "9km (downscaled via elevation lapse rate)",
        "cadence": "Hourly / Daily summary",
        "nominal_latency_days": 5,
        "description": "Global atmospheric reanalysis combining model data with ground weather station assimilations.",
        "type": "Numerical Climate Model"
    },
    "telemetry_live": {
        "name": "Weather-Model-Based Conditions [Estimated]",
        "source": "Open-Meteo API (NOAA GFS / DWD ICON blend)",
        "spatial_resolution": "Numerical model grid cell interpolated to municipal coordinates",
        "cadence": "Hourly forecast cycle",
        "nominal_latency_hours": 1,
        "description": "Atmospheric model output (2m dry-bulb temperature, relative humidity, wind velocity, and apparent heat index). Ambient contextual feed; not applied to static LST spatial predictions.",
        "type": "Weather-Model-Based Conditions [Estimated]"
    },
    "demographics": {
        "name": "Population & Age Demographics",
        "source": "WorldPop 2024 + Census of India / TNSDMA Projections",
        "spatial_resolution": "100m constrained raster",
        "cadence": "Annual / Decennial projection",
        "nominal_latency_days": 365,
        "description": "High-resolution dasymetric population modeling disaggregated by age cohorts (<5, >65) and household income proxies.",
        "type": "Socio-Demographic Census / Model"
    },
    "infrastructure": {
        "name": "Urban Infrastructure & Amenities",
        "source": "OpenStreetMap (OSM) Contributors & Corporation GIS Cell",
        "spatial_resolution": "Vector points & polygons",
        "cadence": "Continuous / Monthly audit",
        "nominal_latency_days": 30,
        "description": "Geolocated public hospitals, primary health centers (UPHCs), parks, water bodies, and arterial road networks.",
        "type": "Vector GIS Layer"
    }
}

# ==============================================================================
# PHYSICAL BOUNDS FOR WHAT-IF SCENARIOS (MODEL DOMAIN OF VALIDITY)
# Prevents unscientific or impossible feature extrapolation
# ==============================================================================
FEATURE_PHYSICAL_BOUNDS = {
    "NDVI": {
        "min": -0.20,
        "max": 0.85,
        "step": 0.02,
        "unit": "index (-1 to 1)",
        "label": "Normalized Difference Vegetation Index (NDVI)",
        "intervention_guidance": "Urban greening can realistically increase NDVI by +0.05 to +0.25."
    },
    "NDBI": {
        "min": -0.40,
        "max": 0.60,
        "step": 0.02,
        "unit": "index (-1 to 1)",
        "label": "Normalized Difference Built-up Index (NDBI)",
        "intervention_guidance": "Cool roofs and de-paving reduce NDBI by -0.05 to -0.20."
    },
    "Albedo": {
        "min": 0.05,
        "max": 0.55,
        "step": 0.01,
        "unit": "reflectance (0 to 1)",
        "label": "Surface Albedo (Reflectance)",
        "intervention_guidance": "High-albedo SRI paint retrofits increase surface albedo from ~0.15 to ~0.35+."
    },
    "Building_Density_pct": {
        "min": 0.0,
        "max": 95.0,
        "step": 1.0,
        "unit": "%",
        "label": "Building Footprint Density",
        "intervention_guidance": "Targeted de-paving and courtyard greening can lower effective density by 5-15%."
    },
    "Dist_to_Water_m": {
        "min": 10.0,
        "max": 8000.0,
        "step": 25.0,
        "unit": "meters",
        "label": "Distance to Water Body",
        "intervention_guidance": "Creating retention ponds or restoring urban canals brings water bodies closer."
    },
    "Dist_to_Park_m": {
        "min": 10.0,
        "max": 5000.0,
        "step": 25.0,
        "unit": "meters",
        "label": "Distance to Urban Green Space",
        "intervention_guidance": "Pocket parks (Miyawaki) establish cooling spaces within 100-300m of dense wards."
    }
}

# ==============================================================================
# MUNICIPAL GOVERNANCE WORKFLOW & HEAT ACTION PLAN (HAP) PROTOCOLS
# ==============================================================================
MUNICIPAL_DEPARTMENTS = [
    "Public Health Department (UPHC)",
    "Engineering & Public Works (PWD)",
    "Parks & Urban Greening Division",
    "Revenue & Disaster Management Authority",
    "Tamil Nadu Water Supply & Drainage (TWAD)",
    "Urban Transport & Traffic Police"
]

INTERVENTION_CATALOG = [
    {
        "id": "cool_roof",
        "name": "Cool Roof Retrofit & Solar Reflective Coating",
        "dept": "Engineering & Public Works (PWD)",
        "cost_per_unit": "₹35 per sq.ft",
        "cooling_potential": "High (-1.5°C to -3.0°C surface LST)",
        "lead_time": "1 - 2 weeks",
        "target_beneficiaries": "Informal settlements, tin/asbestos roof households, schools"
    },
    {
        "id": "urban_forestry",
        "name": "Miyawaki Urban Micro-Forest & Dense Canopy Planting",
        "dept": "Parks & Urban Greening Division",
        "cost_per_unit": "₹45,000 per 100 sq.m pocket",
        "cooling_potential": "High (-1.0°C to -2.5°C localized LST)",
        "lead_time": "3 - 6 months (growth)",
        "target_beneficiaries": "Dense residential neighbourhoods, school zones"
    },
    {
        "id": "hydration_kiosk",
        "name": "Emergency Drinking Water & ORS Hydration Kiosk",
        "dept": "Public Health Department (UPHC)",
        "cost_per_unit": "₹25,000 setup + ₹8,000/mo operating",
        "cooling_potential": "Vulnerability reduction (Heat stroke prevention)",
        "lead_time": "48 hours",
        "target_beneficiaries": "Outdoor daily-wage workers, traffic personnel, gig workers"
    },
    {
        "id": "water_misting",
        "name": "Mobile Water-Misting Tanker Deployment on Corridors",
        "dept": "Engineering & Public Works (PWD)",
        "cost_per_unit": "₹12,000 per tanker day",
        "cooling_potential": "Moderate (-1.0°C ambient wet-bulb cooling during peak noon)",
        "lead_time": "Immediate (24 hours)",
        "target_beneficiaries": "Commercial markets, bus stands, pedestrian corridors"
    },
    {
        "id": "labor_advisory",
        "name": "Outdoor Manual Labor Shift Advisory (12:00 - 15:30 Halt)",
        "dept": "Revenue & Disaster Management Authority",
        "cost_per_unit": "Regulatory enforcement (Zero direct capital cost)",
        "cooling_potential": "Morbidity prevention",
        "lead_time": "Immediate (Official Gazette)",
        "target_beneficiaries": "Construction workers, agricultural laborers, street vendors"
    },
    {
        "id": "shaded_shelters",
        "name": "Cool Shaded Bus Shelters & Public Transit Canopies",
        "dept": "Urban Transport & Traffic Police",
        "cost_per_unit": "₹1,80,000 per transit shelter",
        "cooling_potential": "Thermal comfort enhancement",
        "lead_time": "3 weeks",
        "target_beneficiaries": "Public transit commuters, elderly, schoolchildren"
    }
]

PROPOSAL_STATUSES = ["Draft", "Submitted", "Approved", "Rejected", "Completed"]
URGENCY_LEVELS = ["Immediate Response", "Urgent (1-2 weeks)", "Seasonal HAP (Pre-Summer)", "Long-term Infrastructure"]

# Map Tile Provider Legal Attributions
TILE_ATTRIBUTIONS = {
    "OpenStreetMap": "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors",
    "Esri World Imagery": "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
}
