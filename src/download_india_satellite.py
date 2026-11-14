
import ee
import pandas as pd
import os
import calendar

# ==================================================
# 1. INITIALIZE GOOGLE EARTH ENGINE
# ==================================================

PROJECT_ID = "crop-health-monitoring-514608"

print("Initializing Google Earth Engine...", flush=True)

ee.Initialize(project=PROJECT_ID)

print("Earth Engine initialized successfully!", flush=True)


# ==================================================
# 2. LOAD INDIA CROPLAND LOCATIONS
# ==================================================

input_file = "data/processed/india_30_locations.csv"

locations_df = pd.read_csv(input_file)

print("\nLocations loaded:", len(locations_df), flush=True)

features = []

for _, row in locations_df.iterrows():

    point = ee.Geometry.Point([
        float(row["longitude"]),
        float(row["latitude"])
    ])

    # 100-meter buffer around each sample point
    area = point.buffer(100)

    feature = ee.Feature(area, {
        "location_id": row["location_id"],
        "admin_unit": row["admin_unit"],
        "longitude": float(row["longitude"]),
        "latitude": float(row["latitude"])
    })

    features.append(feature)

locations = ee.FeatureCollection(features)


# ==================================================
# 3. LOAD SENTINEL-2 SATELLITE COLLECTION
# ==================================================

print("\nPreparing Sentinel-2 imagery...", flush=True)

satellite = ee.ImageCollection(
    "COPERNICUS/S2_SR_HARMONIZED"
)


# ==================================================
# 4. CLOUD MASKING
# ==================================================

def mask_clouds(image):

    scl = image.select("SCL")

    # Remove defective pixels, cloud shadows,
    # clouds, cirrus and snow.

    mask = (
        scl.neq(1)
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )

    return image.updateMask(mask).select([
        "B2",
        "B4",
        "B8",
        "B8A",
        "B11"
    ])


# ==================================================
# 5. CALCULATE VEGETATION INDICES
# ==================================================

def calculate_indices(image):

    # Scale reflectance values
    scaled = image.select([
        "B2",
        "B4",
        "B8",
        "B8A",
        "B11"
    ]).multiply(0.0001)

    blue = scaled.select("B2")
    red = scaled.select("B4")
    nir = scaled.select("B8")
    nir_narrow = scaled.select("B8A")
    swir = scaled.select("B11")

    # NDVI
    ndvi = (
        nir.subtract(red)
        .divide(nir.add(red))
        .rename("NDVI")
    )

    # EVI
    evi = (
        nir.subtract(red)
        .multiply(2.5)
        .divide(
            nir.add(red.multiply(6))
            .subtract(blue.multiply(7.5))
            .add(1)
        )
        .rename("EVI")
    )

    # NDMI
    ndmi = (
        nir_narrow.subtract(swir)
        .divide(nir_narrow.add(swir))
        .rename("NDMI")
    )

    return ee.Image.cat([
        ndvi,
        evi,
        ndmi
    ])


# ==================================================
# 6. MONTHLY SATELLITE EXTRACTION
# ==================================================

years = [2023, 2024, 2025]

months = [6, 7, 8, 9, 10]

rows = []

total_months = len(years) * len(months)

processed_months = 0

for year in years:

    for month in months:

        processed_months += 1

        month_name = calendar.month_name[month]

        print(
            f"\n[{processed_months}/{total_months}] "
            f"Processing {month_name} {year}...",
            flush=True
        )

        start_date = f"{year}-{month:02d}-01"

        end_date = (
            f"{year}-{month + 1:02d}-01"
            if month < 12
            else f"{year + 1}-01-01"
        )

        # Filter satellite images
        collection = (
            satellite
            .filterDate(start_date, end_date)
            .filterBounds(locations.geometry())
            .filter(
                ee.Filter.lt(
                    "CLOUDY_PIXEL_PERCENTAGE",
                    80
                )
            )
            .map(mask_clouds)
        )

        # Get image count
        image_count = collection.size().getInfo()

        print(
            "Satellite images found:",
            image_count,
            flush=True
        )

        # If no images exist, retain missing records
        if image_count == 0:

            for _, location in locations_df.iterrows():

                rows.append({
                    "location_id": location["location_id"],
                    "admin_unit": location["admin_unit"],
                    "longitude": location["longitude"],
                    "latitude": location["latitude"],
                    "year": year,
                    "month_number": month,
                    "month_name": month_name,
                    "image_count": 0,
                    "mean_ndvi": None,
                    "mean_evi": None,
                    "mean_ndmi": None
                })

            print("No images available for this month.")

            continue

        # Create monthly median composite
        composite = collection.median()

        # Calculate NDVI, EVI and NDMI
        indices = calculate_indices(composite)

        # Extract average values for each location
        monthly_stats = indices.reduceRegions(
            collection=locations,
            reducer=ee.Reducer.mean(),
            scale=20,
            tileScale=4
        )

        # Retrieve only the current month's results
        print(
            "Retrieving monthly results...",
            flush=True
        )

        result = monthly_stats.getInfo()

        for feature in result["features"]:

            properties = feature["properties"]

            rows.append({
                "location_id": properties.get("location_id"),
                "admin_unit": properties.get("admin_unit"),
                "longitude": properties.get("longitude"),
                "latitude": properties.get("latitude"),
                "year": year,
                "month_number": month,
                "month_name": month_name,
                "image_count": image_count,
                "mean_ndvi": properties.get("NDVI"),
                "mean_evi": properties.get("EVI"),
                "mean_ndmi": properties.get("NDMI")
            })

        print(
            f"Completed {month_name} {year}. "
            f"Records collected: {len(rows)}",
            flush=True
        )


# ==================================================
# 7. CREATE FINAL DATASET
# ==================================================

print(
    "\nCreating India-wide satellite dataset...",
    flush=True
)

df = pd.DataFrame(rows)

os.makedirs(
    "data/processed",
    exist_ok=True
)

output_file = (
    "data/processed/india_satellite_indices.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ==================================================
# 8. DATA QUALITY CHECKS
# ==================================================

print("\n========================================")
print("INDIA SATELLITE EXTRACTION COMPLETED")
print("========================================")

print("\nDataset shape:", df.shape)

print(
    "\nUnique locations:",
    df["location_id"].nunique()
)

print(
    "\nAdministrative units:",
    df["admin_unit"].nunique()
)

print("\nMissing values:")

print(
    df[
        ["mean_ndvi", "mean_evi", "mean_ndmi"]
    ].isnull().sum()
)

print("\nRecords by year:")

print(
    df.groupby("year").size()
)

print("\nRecords by administrative unit:")

print(
    df.groupby("admin_unit").size()
)

print("\nFirst 10 records:")

print(
    df.head(10).to_string(index=False)
)

print("\nSaved successfully:", output_file)

print("\nTotal records:", len(df))
