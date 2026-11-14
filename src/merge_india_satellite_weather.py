
import pandas as pd

# ==========================================
# 1. LOAD DATASETS
# ==========================================

satellite = pd.read_csv(
    "data/processed/india_satellite_indices.csv"
)

weather = pd.read_csv(
    "data/processed/india_monthly_weather.csv"
)

print("Satellite dataset:", satellite.shape)
print("Weather dataset:", weather.shape)


# ==========================================
# 2. FILTER MATCHING MONTHS
# ==========================================

# Keep June, July, August, September and October
weather = weather[
    weather["month_number"].isin([6, 7, 8, 9, 10])
].copy()

print("\nFiltered weather dataset:", weather.shape)


# ==========================================
# 3. SELECT WEATHER FEATURES
# ==========================================

weather_columns = [
    "location_id",
    "year",
    "month_number",
    "total_rainfall",
    "average_temperature",
    "maximum_temperature",
    "minimum_temperature",
    "rainy_days",
    "dry_days",
    "longest_dry_spell"
]

weather = weather[weather_columns]


# ==========================================
# 4. MERGE DATASETS
# ==========================================

merged = pd.merge(
    satellite,
    weather,
    on=["location_id", "year", "month_number"],
    how="left",
    validate="one_to_one"
)


# ==========================================
# 5. CHECK DATA QUALITY
# ==========================================

print("\nMerged dataset shape:", merged.shape)

print(
    "\nUnique locations:",
    merged["location_id"].nunique()
)

print(
    "\nObservations by year:"
)

print(merged.groupby("year").size())

print("\nMissing values:")

print(merged.isnull().sum())

print(
    "\nValid satellite observations:",
    merged["mean_ndvi"].notna().sum()
)

print(
    "\nMissing satellite observations:",
    merged["mean_ndvi"].isna().sum()
)

print(
    "\nMissing rainfall observations:",
    merged["total_rainfall"].isna().sum()
)


# ==========================================
# 6. SAVE FINAL DATASET
# ==========================================

output_file = (
    "data/processed/india_crop_health_weather.csv"
)

merged.to_csv(output_file, index=False)

print("\nFirst 10 records:")

print(merged.head(10).to_string(index=False))

print("\nSaved:", output_file)
