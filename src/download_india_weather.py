
import requests
import pandas as pd
import os
import time
from datetime import timedelta

# ==================================================
# 1. CONFIGURATION
# ==================================================

INPUT_FILE = "data/processed/india_30_locations.csv"

OUTPUT_FILE = "data/processed/india_monthly_weather.csv"

BASE_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
)

PARAMETERS = "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN"

START_DATE = "20230601"
END_DATE = "20251031"

print("INDIA-WIDE WEATHER DATA COLLECTION")
print("=" * 45)


# ==================================================
# 2. LOAD LOCATIONS
# ==================================================

locations = pd.read_csv(INPUT_FILE)

print("Locations loaded:", len(locations))


# ==================================================
# 3. DOWNLOAD DAILY WEATHER
# ==================================================

daily_data = []

session = requests.Session()

for index, location in locations.iterrows():

    location_id = location["location_id"]
    admin_unit = location["admin_unit"]

    latitude = float(location["latitude"])
    longitude = float(location["longitude"])

    print(
        f"\n[{index + 1}/{len(locations)}] "
        f"Downloading weather for {location_id} "
        f"({admin_unit})...",
        flush=True
    )

    params = {
        "parameters": PARAMETERS,
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON",
        "time-standard": "UTC"
    }

    response = None

    # Retry failed requests
    for attempt in range(3):

        try:

            response = session.get(
                BASE_URL,
                params=params,
                timeout=90
            )

            response.raise_for_status()

            break

        except requests.RequestException as error:

            print(
                f"Request failed (attempt {attempt + 1}/3): {error}",
                flush=True
            )

            if attempt == 2:
                raise

            time.sleep(5 * (attempt + 1))

    data = response.json()

    # Extract weather parameters
    weather = data["properties"]["parameter"]

    rainfall = weather["PRECTOTCORR"]
    temperature = weather["T2M"]
    maximum_temperature = weather["T2M_MAX"]
    minimum_temperature = weather["T2M_MIN"]

    # Convert daily values into a DataFrame
    df = pd.DataFrame({
        "date": list(rainfall.keys()),
        "rainfall": list(rainfall.values()),
        "temperature": list(temperature.values()),
        "maximum_temperature": list(maximum_temperature.values()),
        "minimum_temperature": list(minimum_temperature.values())
    })

    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y%m%d"
    )

    # Convert fill values to missing values
    weather_columns = [
        "rainfall",
        "temperature",
        "maximum_temperature",
        "minimum_temperature"
    ]

    for column in weather_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        df.loc[df[column] <= -900, column] = float("nan")

    df["location_id"] = location_id
    df["admin_unit"] = admin_unit
    df["latitude"] = latitude
    df["longitude"] = longitude

    daily_data.append(df)

    print(
        "Daily records downloaded:",
        len(df),
        flush=True
    )

    # Avoid sending requests too rapidly
    time.sleep(1)


# ==================================================
# 4. COMBINE DAILY WEATHER DATA
# ==================================================

print("\nCombining daily weather data...", flush=True)

daily_df = pd.concat(
    daily_data,
    ignore_index=True
)

daily_df["year"] = daily_df["date"].dt.year
daily_df["month_number"] = daily_df["date"].dt.month

print("Combined daily dataset:", daily_df.shape)


# ==================================================
# 5. CALCULATE LONGEST DRY SPELL
# ==================================================

def calculate_longest_dry_spell(rainfall):

    dry = rainfall.lt(1.0).fillna(False)

    longest = 0
    current = 0

    for value in dry:

        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


# ==================================================
# 6. CREATE MONTHLY WEATHER FEATURES
# ==================================================

print("\nCalculating monthly weather features...", flush=True)

monthly_rows = []

for (location_id, year, month), group in daily_df.groupby(
    ["location_id", "year", "month_number"]
):

    group = group.sort_values("date")

    location = group.iloc[0]

    monthly_rows.append({

        "location_id": location_id,

        "admin_unit": location["admin_unit"],

        "latitude": location["latitude"],

        "longitude": location["longitude"],

        "year": year,

        "month_number": month,

        "month_name": group["date"].iloc[0].strftime("%B"),

        "total_rainfall": group["rainfall"].sum(min_count=1),

        "average_temperature": group["temperature"].mean(),

        "maximum_temperature": group["maximum_temperature"].mean(),

        "minimum_temperature": group["minimum_temperature"].mean(),

        "rainy_days": group["rainfall"].ge(1.0).sum(),

        "dry_days": group["rainfall"].lt(1.0).sum(),

        "longest_dry_spell": calculate_longest_dry_spell(
            group["rainfall"]
        )

    })


monthly_df = pd.DataFrame(monthly_rows)


# ==================================================
# 7. SAVE MONTHLY DATASET
# ==================================================

os.makedirs("data/processed", exist_ok=True)

monthly_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==================================================
# 8. DISPLAY RESULTS
# ==================================================

print("\n========================================")
print("INDIA WEATHER EXTRACTION COMPLETED")
print("========================================")

print("\nDaily dataset shape:", daily_df.shape)

print("Monthly dataset shape:", monthly_df.shape)

print(
    "\nUnique locations:",
    monthly_df["location_id"].nunique()
)

print(
    "\nAdministrative units:",
    monthly_df["admin_unit"].nunique()
)

print("\nMissing values:")

print(monthly_df.isnull().sum())

print("\nFirst 10 monthly records:")

print(
    monthly_df.head(10).to_string(index=False)
)

print("\nSaved:", OUTPUT_FILE)
