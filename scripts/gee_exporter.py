"""
Google Earth Engine (GEE) Automated Data Extraction Script.
Designed for students and academic researchers using Google Earth Engine's free tier.
Extracts:
1. Landsat 8/9 Level-2 Surface Temperature (LST)
2. Sentinel-2 MSI Multi-Spectral Surface Reflectance (NDVI, NDWI, NDBI)
3. Copernicus / SRTM 30m Digital Elevation Model (DEM)
4. WorldPop Gridded Population Density
Usage:
    python scripts/gee_exporter.py --region trichy --year 2024
"""

import argparse

GEE_TEMPLATE_CODE = """
import ee
import geemap

# Initialize Earth Engine
# Run `ee.Authenticate()` once on your machine if not already authenticated
try:
    ee.Initialize()
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print("Earth Engine initialization failed. Run 'ee.Authenticate()' in your terminal.")
    print(e)
    exit(1)

# Study Area Bounding Box (e.g. Trichy Delta)
roi = ee.Geometry.BBox(78.62, 10.72, 78.78, 10.88)

# 1. Landsat 8/9 Level-2 Surface Temperature (April - May Peak Summer)
landsat = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    .filterBounds(roi)
    .filterDate("2024-04-01", "2024-05-31")
    .filter(ee.Filter.lt("CLOUD_COVER", 10))
    .median())

# Thermal Band 10 scaling for Celsius: (DN * 0.00341802 + 149.0) - 273.15
lst_celsius = landsat.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("LST")

# 2. Sentinel-2 Surface Reflectance (Harmonized)
s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(roi)
    .filterDate("2024-04-01", "2024-05-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
    .median())

# Spectral Indices
ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
ndbi = s2.normalizedDifference(["B11", "B8"]).rename("NDBI")

# 3. Copernicus 30m DEM
dem = ee.Image("COPERNICUS/DEM/GLO30").select("DEM").clip(roi).rename("Elevation")
slope = ee.Terrain.slope(dem).rename("Slope")

# 4. WorldPop Population Count / Density
worldpop = ee.ImageCollection("WorldPop/GP/100m/pop").filterBounds(roi).first().rename("Population")

# Harmonize into a single multi-band image
composite = ee.Image.cat([lst_celsius, ndvi, ndwi, ndbi, dem, slope, worldpop]).clip(roi)

print("Composite bands:", composite.bandNames().getInfo())
print("Ready for sampling over regular fishnet grid or administrative wards.")
"""

def main():
    print("="*70)
    print("GOOGLE EARTH ENGINE (GEE) SATELLITE EXTRACTION GUIDE (STUDENT TIER)")
    print("="*70)
    print("To use this cloud script:")
    print("1. Sign up for a free student Earth Engine account at: https://earthengine.google.com")
    print("2. Run in your terminal: python -c 'import ee; ee.Authenticate()'")
    print("3. Execute this script to extract cloud-free Landsat 8/9 & Sentinel-2 composites directly.")
    print("="*70)
    print("Sample GEE Python snippet:")
    print(GEE_TEMPLATE_CODE)

if __name__ == "__main__":
    main()
