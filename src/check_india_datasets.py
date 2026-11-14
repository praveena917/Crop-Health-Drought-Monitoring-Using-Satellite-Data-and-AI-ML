
import ee

PROJECT_ID = "crop-health-monitoring-514608"

ee.Initialize(project=PROJECT_ID)

# Load 2025 administrative boundaries
all_units = ee.FeatureCollection(
    "FAO/GAUL/2025/level1"
)

# Correct country property for GAUL 2025
india_states = all_units.filter(
    ee.Filter.eq("GAUL0_NAME", "India")
)

# Check administrative units
unit_count = india_states.size().getInfo()

print("INDIA-WIDE DATASET CHECK")
print("------------------------")
print("Administrative units:", unit_count)

if unit_count == 0:
    print("ERROR: India boundaries were not found.")
    raise SystemExit

# Get state and UT names
state_names = india_states.aggregate_array(
    "GAUL1_NAME"
).getInfo()

print("\nAdministrative units found:")
for name in state_names:
    print(name)

# India geometry
india_geometry = india_states.geometry()

# Load ESA WorldCover 2021
worldcover = ee.ImageCollection(
    "ESA/WorldCover/v200"
).first()

# Cropland class = 40
cropland = worldcover.select("Map").eq(40)

# Check cropland distribution
cropland_stats = cropland.reduceRegion(
    reducer=ee.Reducer.frequencyHistogram(),
    geometry=india_geometry,
    scale=10000,
    maxPixels=1e13,
    bestEffort=True
).getInfo()

print("\nCropland classification histogram:")
print(cropland_stats)

print("\nAdministrative boundary and cropland checks completed.")
