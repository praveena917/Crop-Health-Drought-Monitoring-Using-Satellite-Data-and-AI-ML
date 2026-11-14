
import pandas as pd

FILE = "data/processed/india_crop_health_weather.csv"

df = pd.read_csv(FILE)

print("\nINDIA INTEGRATED DATASET VALIDATION")
print("=" * 50)

# 1. Dataset overview
print("\nDataset shape:", df.shape)
print("Unique locations:", df["location_id"].nunique())

# 2. Duplicate observations
duplicates = df.duplicated(
    subset=["location_id", "year", "month_number"]
).sum()

print("\nDuplicate location-month records:", duplicates)

# 3. Satellite data ranges
print("\nSatellite index statistics:")
print(
    df[["mean_ndvi", "mean_evi", "mean_ndmi"]]
    .describe()
    .round(3)
)

# 4. Check suspicious index values
for column in ["mean_ndvi", "mean_evi", "mean_ndmi"]:

    invalid = (
        (df[column] < -1) |
        (df[column] > 1)
    ).sum()

    print(f"{column} values outside [-1, 1]:", invalid)

# 5. Weather statistics
print("\nWeather statistics:")
print(
    df[
        [
            "total_rainfall",
            "average_temperature",
            "maximum_temperature",
            "minimum_temperature"
        ]
    ].describe().round(2)
)

# 6. Valid satellite observations by year
print("\nValid satellite observations by year:")

print(
    df.groupby("year")["mean_ndvi"]
    .agg(["count", "mean"])
    .round(3)
)

# 7. Valid observations by location
print("\nValid satellite observations by location:")

print(
    df.groupby(["location_id", "admin_unit"])["mean_ndvi"]
    .agg(["count", "mean"])
    .round(3)
    .to_string()
)

# 8. Weather consistency
invalid_weather = (
    (df["total_rainfall"] < 0) |
    (df["maximum_temperature"] < df["minimum_temperature"]) |
    (df["average_temperature"] < df["minimum_temperature"]) |
    (df["average_temperature"] > df["maximum_temperature"])
).sum()

print("\nSuspicious weather records:", invalid_weather)

# 9. Final missing values
print("\nMissing values:")
print(df.isnull().sum())

print("\nVALIDATION COMPLETED")
