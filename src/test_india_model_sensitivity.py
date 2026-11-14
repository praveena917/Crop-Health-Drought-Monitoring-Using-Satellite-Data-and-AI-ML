
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest

INPUT_FILE = "data/processed/india_drought_features.csv"
SCALER_FILE = "models/india_scaler.pkl"

FEATURES = [
    "mean_ndvi",
    "mean_evi",
    "mean_ndmi",
    "total_rainfall",
    "average_temperature",
    "rainy_days",
    "dry_days",
    "longest_dry_spell"
]

print("MODEL SENSITIVITY ANALYSIS")
print("=" * 45)

df = pd.read_csv(INPUT_FILE)

model_df = df.dropna(subset=FEATURES).copy()

X = model_df[FEATURES]

scaler = joblib.load(SCALER_FILE)

X_scaled = scaler.transform(X)

results = {}

for contamination in [0.05, 0.10, 0.15]:

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )

    predictions = model.fit_predict(X_scaled)

    flagged = set(model_df.index[predictions == -1])

    results[contamination] = flagged

    print(
        f"\nContamination: {contamination:.0%}"
    )

    print("Unusual observations:", len(flagged))

    print(
        model_df.loc[
            list(flagged)
        ].groupby("year").size()
    )

# Compare overlap
print("\nOVERLAP BETWEEN SETTINGS")

for value in [0.05, 0.10, 0.15]:

    print(
        f"{value:.0%} unusual observations also flagged "
        f"at 10%: ",
        len(results[value] & results[0.10])
    )

print("\nOBSERVATIONS FLAGGED AT ALL THREE SETTINGS:")

common = (
    results[0.05] &
    results[0.10] &
    results[0.15]
)

print("Count:", len(common))

print(
    model_df.loc[
        sorted(common),
        [
            "location_id",
            "admin_unit",
            "year",
            "month_name",
            "mean_ndvi",
            "mean_ndmi",
            "total_rainfall"
        ]
    ].to_string(index=False)
)

print("\nSENSITIVITY ANALYSIS COMPLETED!")
