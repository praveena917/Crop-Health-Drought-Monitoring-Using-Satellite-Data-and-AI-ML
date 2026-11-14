
import pandas as pd
import os

INPUT_FILE = "data/processed/india_crop_health_weather.csv"

OUTPUT_FILE = "data/processed/india_drought_features.csv"

print("INDIA-WIDE DROUGHT FEATURE ENGINEERING")
print("=" * 50)

# Load dataset
df = pd.read_csv(INPUT_FILE)

df = df.sort_values(
    ["location_id", "year", "month_number"]
).reset_index(drop=True)

# 1. Temperature range
df["temperature_range"] = (
    df["maximum_temperature"] -
    df["minimum_temperature"]
)

# 2. Three-month rolling rainfall and dry days
df["rainfall_3month_total"] = (
    df.groupby(["location_id", "year"])["total_rainfall"]
    .transform(lambda x: x.rolling(3, min_periods=3).sum())
)

df["dry_days_3month"] = (
    df.groupby(["location_id", "year"])["dry_days"]
    .transform(lambda x: x.rolling(3, min_periods=3).sum())
)

# 3. Month-to-month vegetation changes
group = df.groupby(["location_id", "year"])

previous_month = group["month_number"].shift(1)

consecutive_months = (
    df["month_number"] - previous_month
).eq(1)

previous_ndvi = group["mean_ndvi"].shift(1)
previous_ndmi = group["mean_ndmi"].shift(1)

df["ndvi_change"] = (
    df["mean_ndvi"] - previous_ndvi
).where(consecutive_months)

df["ndmi_change"] = (
    df["mean_ndmi"] - previous_ndmi
).where(consecutive_months)

# 4. Save the engineered dataset
os.makedirs("data/processed", exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

# 5. Report
print("\nOriginal dataset shape:", (len(df), 18))
print("Feature-engineered dataset shape:", df.shape)

print("\nNew features:")
print([
    "temperature_range",
    "rainfall_3month_total",
    "dry_days_3month",
    "ndvi_change",
    "ndmi_change"
])

print("\nMissing values in new features:")

new_features = [
    "temperature_range",
    "rainfall_3month_total",
    "dry_days_3month",
    "ndvi_change",
    "ndmi_change"
]

print(df[new_features].isnull().sum())

print("\nSample records:")

print(
    df[
        [
            "location_id",
            "year",
            "month_name",
            "mean_ndvi",
            "mean_ndmi",
            "total_rainfall",
            "rainfall_3month_total",
            "dry_days_3month",
            "temperature_range",
            "ndvi_change",
            "ndmi_change"
        ]
    ].head(15).to_string(index=False)
)

print("\nSaved:", OUTPUT_FILE)
print("\nFEATURE ENGINEERING COMPLETED!")
