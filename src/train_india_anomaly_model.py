
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

INPUT_FILE = "data/processed/india_drought_features.csv"

OUTPUT_FILE = "data/processed/india_anomaly_results.csv"

MODEL_FILE = "models/india_isolation_forest.pkl"

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

print("INDIA-WIDE ANOMALY DETECTION")
print("=" * 50)

# 1. Load data
df = pd.read_csv(INPUT_FILE)

# 2. Select complete observations
model_df = df.dropna(subset=FEATURES).copy()

X = model_df[FEATURES]

print("\nTotal observations:", len(df))
print("Observations used:", len(model_df))
print("Features:", len(FEATURES))

# 3. Scale features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# 4. Train Isolation Forest
model = IsolationForest(
    n_estimators=200,
    contamination=0.10,
    random_state=42
)

model.fit(X_scaled)

# 5. Predict anomalies
model_df["anomaly_prediction"] = model.predict(X_scaled)

model_df["anomaly_score"] = model.decision_function(X_scaled)

model_df["environmental_status"] = np.where(
    model_df["anomaly_prediction"] == -1,
    "Unusual",
    "Normal"
)

# 6. Save model and scaler
os.makedirs("models", exist_ok=True)

joblib.dump(model, MODEL_FILE)

joblib.dump(scaler, SCALER_FILE)

# 7. Save results
model_df.to_csv(OUTPUT_FILE, index=False)

# 8. Display results
print("\nEnvironmental status distribution:")

print(model_df["environmental_status"].value_counts())

print("\nUnusual observations by year:")

print(
    model_df[model_df["environmental_status"] == "Unusual"]
    .groupby("year")
    .size()
)

print("\nUnusual observations by administrative unit:")

print(
    model_df[model_df["environmental_status"] == "Unusual"]
    .groupby("admin_unit")
    .size()
)

print("\nFirst 10 unusual observations:")

print(
    model_df[
        model_df["environmental_status"] == "Unusual"
    ][
        [
            "location_id",
            "admin_unit",
            "year",
            "month_name",
            "mean_ndvi",
            "total_rainfall",
            "anomaly_score"
        ]
    ].head(10).to_string(index=False)
)

print("\nSaved model:", MODEL_FILE)
print("Saved scaler:", SCALER_FILE)
print("Saved results:", OUTPUT_FILE)

print("\nMODEL TRAINING COMPLETED!")
