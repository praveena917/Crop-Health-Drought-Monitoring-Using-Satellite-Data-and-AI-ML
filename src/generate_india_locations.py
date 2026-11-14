
import ee
import pandas as pd
import os

# --------------------------------------------------
# 1. Initialize Google Earth Engine
# --------------------------------------------------

PROJECT_ID = "crop-health-monitoring-514608"

print("Initializing Google Earth Engine...", flush=True)

ee.Initialize(project=PROJECT_ID)

print("Earth Engine initialized successfully!", flush=True)


# --------------------------------------------------
# 2. Load India's administrative boundaries
# --------------------------------------------------

print("\nLoading India administrative boundaries...", flush=True)

states = (
    ee.FeatureCollection("FAO/GAUL/2025/level1")
    .filter(ee.Filter.eq("GAUL0_NAME", "India"))
)

# Exclude disputed boundary entries
all_names = states.aggregate_array("GAUL1_NAME").getInfo()

valid_names = [
    name for name in all_names
    if "Disputed" not in name
]

states = states.filter(
    ee.Filter.inList("GAUL1_NAME", valid_names)
)

india_geometry = states.geometry()

print("Administrative units loaded:", len(valid_names), flush=True)


# --------------------------------------------------
# 3. Load ESA WorldCover cropland data
# --------------------------------------------------

print("\nLoading ESA WorldCover cropland data...", flush=True)

worldcover = ee.ImageCollection(
    "ESA/WorldCover/v200"
).first()

# ESA WorldCover class 40 = Cropland
cropland = worldcover.select("Map").eq(40).selfMask()


# --------------------------------------------------
# 4. Generate exactly 30 cropland sample locations
# --------------------------------------------------

print("\nSampling 30 cropland locations...", flush=True)

samples = cropland.stratifiedSample(
    numPoints=30,
    classBand="Map",
    region=india_geometry,
    scale=1000,
    classValues=[1],
    classPoints=[30],
    seed=42,
    geometries=True,
    tileScale=4
)

print("Cropland sampling completed!", flush=True)


# --------------------------------------------------
# 5. Attach administrative unit names
# --------------------------------------------------

print("\nMatching locations to administrative units...", flush=True)

def add_admin(feature):

    matched_state = states.filterBounds(
        feature.geometry()
    ).first()

    return feature.set(
        "admin_unit",
        ee.Feature(matched_state).get("GAUL1_NAME")
    )


samples = samples.map(add_admin)


# --------------------------------------------------
# 6. Retrieve sampled locations
# --------------------------------------------------

print("\nRetrieving location data...", flush=True)

result = samples.getInfo()

rows = []

for i, feature in enumerate(result["features"]):

    coordinates = feature["geometry"]["coordinates"]
    properties = feature["properties"]

    rows.append({
        "location_id": f"India_{i+1:02d}",
        "admin_unit": properties["admin_unit"],
        "longitude": coordinates[0],
        "latitude": coordinates[1],
        "land_cover_class": properties["Map"]
    })


# --------------------------------------------------
# 7. Create and save dataset
# --------------------------------------------------

df = pd.DataFrame(rows)

os.makedirs("data/processed", exist_ok=True)

output_file = "data/processed/india_30_locations.csv"

df.to_csv(output_file, index=False)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("\n======================================")
print("INDIA CROPLAND SAMPLING COMPLETED")
print("======================================")

print("\nDataset shape:", df.shape)

print("\nTotal locations generated:", len(df))

print("\nLocations by administrative unit:")

print(df["admin_unit"].value_counts().to_string())

print("\nFirst 10 locations:")

print(df.head(10).to_string(index=False))

print("\nSaved successfully:", output_file)

if len(df) != 30:
    print("\nWARNING: Expected 30 locations, but generated", len(df))
