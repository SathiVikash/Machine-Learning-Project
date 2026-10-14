"""
Data Generator module for Urban Heat Risk Assessment.
Generates realistic, physically consistent geospatial datasets and administrative wards
for Tiruchirappalli (Cauvery Delta) and Chennai (Non-Delta Coastal Metro).
"""

import numpy as np
import pandas as pd
import json
from shapely.geometry import Polygon, MultiPolygon, Point, box
from shapely.ops import unary_union
from src.config import REGIONS, RAW_DATA_DIR, PROCESSED_DATA_DIR

def generate_synthetic_region_data(region_key="trichy_delta", grid_size_m=250, random_seed=42):
    """
    Generates realistic spatial grid points and synthetic remote sensing/urban/demographic
    indicators for the selected region.
    """
    np.random.seed(random_seed)
    region = REGIONS[region_key]
    bounds = region["bounds"]
    
    # Generate spatial coordinates
    # Approximate degree conversion: 1 deg lat ~ 111,000m, 1 deg lon ~ 111,000 * cos(lat)
    lat_step = grid_size_m / 111000.0
    lon_step = grid_size_m / (111000.0 * np.cos(np.radians(region["center_lat"])))
    
    lats = np.arange(bounds["min_lat"], bounds["max_lat"], lat_step)
    lons = np.arange(bounds["min_lon"], bounds["max_lon"], lon_step)
    
    grid_points = []
    grid_polygons = []
    
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            pt = Point(lon, lat)
            poly = box(lon - lon_step/2, lat - lat_step/2, lon + lon_step/2, lat + lat_step/2)
            grid_points.append({
                "grid_id": f"GRID_{region_key}_{i:03d}_{j:03d}",
                "lat": lat,
                "lon": lon,
                "poly": poly
            })
            
    df = pd.DataFrame(grid_points)
    n_points = len(df)
    
    center_lat = region["center_lat"]
    center_lon = region["center_lon"]
    
    # Distance to city center (km)
    dist_center = np.sqrt(
        ((df["lat"] - center_lat) * 111.0)**2 +
        ((df["lon"] - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2
    )
    
    if region_key == "trichy_delta":
        # Cauvery River corridor runs approximately along lat 10.85 to 10.87 east-west
        dist_cauvery = np.abs((df["lat"] - (10.860 + 0.05 * (df["lon"] - 78.70))) * 111.0)
        # Srirangam Island between Cauvery and Kollidam
        is_srirangam = (df["lat"] > 10.84) & (df["lat"] < 10.875) & (df["lon"] > 78.67) & (df["lon"] < 78.72)
        
        # Delta agricultural zones (fertile fringe to the east and north-east)
        is_delta_fringe = (df["lon"] > 78.72) & (df["lat"] > 10.80)
        
        # Industrial cluster towards South-East (BHEL / Thuvakudi corridor)
        dist_industrial = np.sqrt(((df["lat"] - 10.76)**2 + (df["lon"] - 78.77)**2)) * 111.0
        
        # Spectral Indices
        # Water index (high near river)
        ndwi_base = 0.65 * np.exp(-dist_cauvery / 0.4) - 0.35
        df["NDWI"] = np.clip(ndwi_base + np.random.normal(0, 0.05, n_points), -0.5, 0.8)
        
        # Vegetation index (high in delta fringe and Srirangam island)
        ndvi_base = 0.22 + 0.35 * np.exp(-dist_center / 3.0) * -1 + 0.35 * np.exp(-dist_cauvery / 1.5)
        ndvi_base[is_srirangam] += 0.25
        ndvi_base[is_delta_fringe] += 0.20
        df["NDVI"] = np.clip(ndvi_base + np.random.normal(0, 0.06, n_points), -0.1, 0.75)
        
        # Built-up index (high in center and industrial zone)
        ndbi_base = 0.45 * np.exp(-dist_center / 3.5) + 0.35 * np.exp(-dist_industrial / 2.5) - 0.2
        ndbi_base[df["NDWI"] > 0.1] = -0.35
        df["NDBI"] = np.clip(ndbi_base + np.random.normal(0, 0.05, n_points), -0.45, 0.65)
        
        # Urban Form
        df["Building_Density_pct"] = np.clip(
            (75.0 * np.exp(-dist_center / 3.2) + 50.0 * np.exp(-dist_industrial / 2.8) - 40.0 * (df["NDWI"] > 0))
            + np.random.normal(0, 5, n_points), 0.0, 92.0
        )
        df["Road_Density_km_km2"] = np.clip(
            (14.0 * np.exp(-dist_center / 4.0) + 10.0 * np.exp(-dist_industrial / 3.5))
            + np.random.normal(0, 1.2, n_points), 0.5, 22.0
        )
        
        df["Dist_to_Water_m"] = np.clip(dist_cauvery * 1000.0 + np.random.uniform(0, 100, n_points), 20, 12000)
        df["Dist_to_Park_m"] = np.clip(dist_center * 400.0 + (1 - df["NDVI"]) * 800.0 + np.random.uniform(50, 300, n_points), 50, 4500)
        
        # Terrain
        df["Elevation_m"] = 85.0 + 20.0 * np.exp(-((df["lat"] - 10.82)**2 + (df["lon"] - 78.69)**2) * 50) + np.random.normal(0, 2, n_points)
        df["Slope_deg"] = np.clip(np.random.exponential(1.5, n_points), 0.0, 15.0)
        
        # Climate Auxiliary (ERA5 representative pre-monsoon summer)
        df["Air_Temp_ERA5"] = 37.5 + np.random.normal(0, 0.4, n_points)
        df["Relative_Humidity_pct"] = 48.0 - 0.5 * (df["Building_Density_pct"] / 10.0) + np.random.normal(0, 1.5, n_points)
        df["Albedo"] = np.clip(0.12 + 0.15 * (df["NDBI"] + 0.45) - 0.08 * df["NDVI"] + np.random.normal(0, 0.02, n_points), 0.08, 0.35)

    else:
        # Chennai: Coastal plain, Bay of Bengal to the east (lon > 80.28)
        dist_coast = np.maximum(0.0, (80.29 - df["lon"]) * 111.0 * np.cos(np.radians(13.08)))
        is_sea = df["lon"] > 80.295
        
        # Major Water bodies: Adyar & Cooum rivers
        dist_river = np.abs((df["lat"] - 13.04) * 111.0)
        
        df["NDWI"] = np.where(is_sea, 0.85, 0.5 * np.exp(-dist_river / 0.5) - 0.3 + np.random.normal(0, 0.04, n_points))
        df["NDVI"] = np.clip(0.18 + 0.25 * (dist_center / 8.0) - 0.25 * is_sea + np.random.normal(0, 0.05, n_points), -0.1, 0.70)
        df["NDBI"] = np.clip(0.55 * np.exp(-dist_center / 6.0) - 0.5 * is_sea + np.random.normal(0, 0.05, n_points), -0.5, 0.70)
        
        df["Building_Density_pct"] = np.clip(
            (82.0 * np.exp(-dist_center / 5.5)) * (~is_sea) + np.random.normal(0, 6, n_points), 0.0, 95.0
        )
        df["Road_Density_km_km2"] = np.clip(
            (18.0 * np.exp(-dist_center / 6.0)) * (~is_sea) + np.random.normal(0, 1.5, n_points), 0.5, 26.0
        )
        df["Dist_to_Water_m"] = np.clip(dist_river * 1000.0, 30, 8000)
        df["Dist_to_Park_m"] = np.clip(dist_center * 350.0 + 200, 50, 4000)
        
        df["Elevation_m"] = np.clip(6.0 + 2.5 * dist_coast + np.random.normal(0, 1.5, n_points), 1.0, 45.0)
        df["Slope_deg"] = np.clip(np.random.exponential(1.0, n_points), 0.0, 8.0)
        
        df["Air_Temp_ERA5"] = 36.8 + np.random.normal(0, 0.3, n_points)
        df["Relative_Humidity_pct"] = 62.0 + 8.0 * np.exp(-dist_coast / 3.0) + np.random.normal(0, 2.0, n_points)
        df["Albedo"] = np.clip(0.14 + 0.12 * (df["NDBI"] + 0.4) - 0.06 * df["NDVI"] + np.random.normal(0, 0.02, n_points), 0.08, 0.35)

    # Physical Target Simulation: Land Surface Temperature (LST in Celsius)
    # Physically derived: higher NDBI and Building Density increase LST,
    # higher NDVI and NDWI decrease LST, proximity to water moderates temperature.
    cooling_vegetation = -9.5 * df["NDVI"]
    cooling_water = -12.0 * np.clip(df["NDWI"], 0.0, 1.0) - 2.5 * np.exp(-df["Dist_to_Water_m"] / 400.0)
    heating_built = 11.0 * np.clip(df["NDBI"], 0.0, 1.0) + 0.08 * df["Building_Density_pct"] + 0.25 * df["Road_Density_km_km2"]
    terrain_effect = -0.0065 * (df["Elevation_m"] - 10.0) # Lapse rate
    albedo_effect = 4.0 * (df["Albedo"] - 0.15)
    residual_noise = np.random.normal(0, 0.75, n_points)
    
    base_temp = 37.5 if region_key == "trichy_delta" else 36.2
    df["LST_Celsius"] = np.round(
        base_temp + cooling_vegetation + cooling_water + heating_built + terrain_effect + albedo_effect + residual_noise,
        2
    )
    # Cap realistic limits for Tamil Nadu peak summer (31.0°C water to 44.5°C dense asphalt)
    df["LST_Celsius"] = np.clip(df["LST_Celsius"], 29.5, 45.0)

    # Demographic & Vulnerability Variables (WorldPop & Municipal Profiles)
    # Dense central core has higher population density & slum percentage
    df["Population_Density_per_km2"] = np.round(
        np.clip(18000 * np.exp(-dist_center / 3.8) + np.random.normal(1200, 800, n_points), 400, 32000), 0
    )
    df["Elderly_Ratio_pct"] = np.round(np.clip(8.5 + 4.0 * np.exp(-dist_center / 5.0) + np.random.normal(0, 1.2, n_points), 4.0, 22.0), 2)
    df["Children_Ratio_pct"] = np.round(np.clip(9.0 + 3.0 * np.sin(df["lat"] * 30) + np.random.normal(0, 1.0, n_points), 4.5, 18.0), 2)
    df["Low_Income_Household_pct"] = np.round(np.clip(18.0 + 15.0 * np.exp(-dist_center / 4.0) + np.random.normal(0, 4.0, n_points), 5.0, 65.0), 1)
    df["Distance_to_Hospital_m"] = np.round(np.clip(dist_center * 600.0 + np.random.uniform(100, 1200, n_points), 150, 8000), 0)

    # Assign Administrative Wards
    num_wards = region["wards_count"]
    # Partition space into ward clusters using spatial grid k-means or regular blocks
    ward_rows = int(np.sqrt(num_wards))
    ward_cols = int(np.ceil(num_wards / ward_rows))
    
    lat_bins = np.linspace(df["lat"].min(), df["lat"].max(), ward_rows + 1)
    lon_bins = np.linspace(df["lon"].min(), df["lon"].max(), ward_cols + 1)
    
    df["ward_row"] = pd.cut(df["lat"], bins=lat_bins, labels=False, include_lowest=True).fillna(0).astype(int)
    df["ward_col"] = pd.cut(df["lon"], bins=lon_bins, labels=False, include_lowest=True).fillna(0).astype(int)
    
    df["ward_num"] = df["ward_row"] * ward_cols + df["ward_col"] + 1
    df["ward_id"] = df["ward_num"].apply(lambda x: f"W_{x:03d}")
    df["ward_name"] = df["ward_num"].apply(lambda x: f"{region['name'][:6].strip()} Ward {x}")
    df["region"] = region_key

    # Save to CSV and JSON
    csv_path = PROCESSED_DATA_DIR / f"{region_key}_grid_features.csv"
    df_export = df.drop(columns=["poly", "ward_row", "ward_col"])
    df_export.to_csv(csv_path, index=False)
    print(f"[{region_key}] Successfully generated {len(df)} grid cells -> {csv_path}")

    # Generate Administrative Ward Boundaries (GeoJSON format)
    wards_geojson = generate_ward_geojson(df, region_key)
    geojson_path = PROCESSED_DATA_DIR / f"{region_key}_wards.geojson"
    with open(geojson_path, "w") as f:
        json.dump(wards_geojson, f)
    print(f"[{region_key}] Successfully generated {num_wards} administrative wards -> {geojson_path}")

    return df_export, wards_geojson

def generate_ward_geojson(df, region_key):
    """
    Creates administrative ward polygons by aggregating grid bounding boxes
    into clean GeoJSON polygons.
    """
    features = []
    unique_wards = df.groupby("ward_id")
    
    for ward_id, group in unique_wards:
        min_lon = group["lon"].min() - 0.0012
        max_lon = group["lon"].max() + 0.0012
        min_lat = group["lat"].min() - 0.0012
        max_lat = group["lat"].max() + 0.0012
        
        ward_poly = Polygon([
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat]
        ])
        
        mean_lst = float(np.round(group["LST_Celsius"].mean(), 2))
        pop_density = float(np.round(group["Population_Density_per_km2"].mean(), 1))
        ward_name = str(group["ward_name"].iloc[0])
        
        features.append({
            "type": "Feature",
            "properties": {
                "ward_id": ward_id,
                "ward_name": ward_name,
                "mean_lst": mean_lst,
                "pop_density": pop_density,
                "region": region_key
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [list(ward_poly.exterior.coords)]
            }
        })
        
    return {
        "type": "FeatureCollection",
        "features": features
    }

if __name__ == "__main__":
    for reg in ["trichy_delta", "chennai_nondelta"]:
        generate_synthetic_region_data(reg)
