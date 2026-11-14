
# India-Wide Crop Health & Drought Monitoring Using Satellite Data and AI/ML

An AI/ML and geospatial analytics project that combines Sentinel-2 satellite imagery, vegetation indices, and NASA POWER weather data to monitor crop health and identify unusual environmental conditions across selected cropland locations in India.

## Project Overview

Agricultural monitoring requires understanding both vegetation conditions and weather patterns. This project integrates satellite-derived vegetation indicators with rainfall and temperature information to explore changes in crop health and detect unusual environmental patterns.

The system uses **Google Earth Engine** for satellite data processing, **NASA POWER** for weather data collection, **Python** for data analysis and feature engineering, and **Isolation Forest** for unsupervised anomaly detection.

An interactive Streamlit dashboard presents the results through maps, charts, filters, and downloadable datasets.

## Key Features

- India-wide sample of 30 cropland locations across 12 administrative units.
- Sentinel-2 satellite data processing using Google Earth Engine.
- Calculation of NDVI, EVI, and NDMI vegetation indices.
- Weather data collection using the NASA POWER API.
- Monthly rainfall and temperature feature generation.
- Satellite and weather dataset integration.
- Exploratory data analysis and data-quality validation.
- Feature engineering for vegetation and weather indicators.
- Isolation Forest-based environmental anomaly detection.
- Interactive Streamlit dashboard with geographical visualization.
- CSV export of filtered monitoring results.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Geospatial Analysis | Google Earth Engine, GeoPandas, Rasterio, Shapely |
| Satellite Data | Sentinel-2 Surface Reflectance Harmonized |
| Weather Data | NASA POWER API |
| Machine Learning | Scikit-learn, Isolation Forest |
| Interactive Dashboard | Streamlit, Folium |
| Development | VS Code, Jupyter Notebook |

## Data Sources

### 1. Sentinel-2 Satellite Imagery

Sentinel-2 Surface Reflectance Harmonized imagery is accessed through Google Earth Engine.

The project calculates three vegetation indicators:

**NDVI — Normalized Difference Vegetation Index**

Measures vegetation greenness and is commonly used to assess vegetation conditions.

\[
NDVI = \frac{B8-B4}{B8+B4}
\]

**EVI — Enhanced Vegetation Index**

Improves vegetation monitoring by accounting for atmospheric and soil-background influences.

\[
EVI = 2.5 \times \frac{B8-B4}{B8+6B4-7.5B2+1}
\]

**NDMI — Normalized Difference Moisture Index**

Provides an indicator of vegetation moisture conditions.

\[
NDMI = \frac{B8A-B11}{B8A+B11}
\]

Cloud and shadow masking is performed using the Sentinel-2 Scene Classification Layer (SCL). Monthly satellite composites are generated for the study period.

### 2. NASA POWER Weather Data

Daily weather data is collected for each of the 30 sample locations.

Parameters include:

- Corrected precipitation (PRECTOTCORR)
- Average temperature (T2M)
- Maximum temperature (T2M_MAX)
- Minimum temperature (T2M_MIN)

Daily weather records are aggregated into monthly features, including:

- Total monthly rainfall
- Average temperature
- Average of daily maximum temperatures
- Average of daily minimum temperatures
- Rainy days
- Dry days
- Longest consecutive dry spell

## Study Period and Dataset

| Dataset component | Details |
|---|---|
| Geographic scope | Selected cropland locations across India |
| Sample locations | 30 |
| Administrative units represented | 12 |
| Study period | June 2023 – October 2025 |
| Satellite observation frequency | Monthly, June–October |
| Expected satellite records | 450 |
| Valid satellite observations | 399 |
| Missing satellite observations | 51 |
| Satellite data completeness | 88.67% |
| Daily weather records | 26,520 |
| Monthly weather records | 870 |

The satellite dataset contains 450 expected location-month records, of which 399 have valid vegetation-index observations. Missing satellite observations are retained in the integrated dataset and excluded from model training when required features are unavailable.

## Project Workflow

### 1. Geospatial Data Preparation

- Identified India administrative units using the FAO GAUL dataset.
- Used ESA WorldCover cropland classification (class 40) to identify candidate cropland locations.
- Generated 30 sample points using Google Earth Engine.
- Stored location identifiers, coordinates, and administrative-unit information.

### 2. Satellite Data Processing

- Accessed Sentinel-2 imagery through Google Earth Engine.
- Applied cloud and shadow masking.
- Generated monthly median satellite composites.
- Calculated NDVI, EVI, and NDMI.
- Extracted vegetation indicators around the selected sample locations.
- Exported the results into structured CSV files.

### 3. Weather Data Collection

- Retrieved daily weather data from the NASA POWER API.
- Collected data separately for all 30 sample coordinates.
- Aggregated daily observations into monthly weather indicators.
- Generated rainfall, temperature, rainy-day, dry-day, and dry-spell features.

### 4. Data Integration and Validation

- Merged satellite vegetation indicators with monthly weather records.
- Checked missing values and duplicate location-month records.
- Validated vegetation-index ranges.
- Checked weather data for missing or suspicious values.
- Investigated missing satellite observations caused by cloud masking or insufficient valid pixels.

### 5. Feature Engineering

Created additional environmental features:

- Temperature range
- Three-month rainfall total
- Three-month dry-day count
- Monthly NDVI change
- Monthly NDMI change

These features support exploratory analysis of vegetation and weather patterns. The current anomaly model uses its selected eight core vegetation and weather features rather than all engineered features.

### 6. Machine Learning — Isolation Forest

An unsupervised Isolation Forest model is used to identify observations with unusual combinations of vegetation and weather characteristics.

**Model configuration:**

| Parameter | Value |
|---|---|
| Algorithm | Isolation Forest |
| Number of estimators | 200 |
| Contamination | 0.10 |
| Random state | 42 |
| Training observations | 399 |
| Input features | 8 |

**Features used for anomaly detection:**

1. Mean NDVI
2. Mean EVI
3. Mean NDMI
4. Total rainfall
5. Average temperature
6. Rainy days
7. Dry days
8. Longest dry spell

The model identifies unusual environmental observations rather than classifying confirmed drought events.

### 7. Interactive Dashboard

The Streamlit dashboard provides:

- Sample-location map of India
- Monitoring summary metrics
- Year and location filters
- NDVI trend visualization
- Monthly rainfall comparison
- Administrative-unit anomaly comparison
- NDVI versus rainfall analysis
- Unusual environmental observations table
- Filtered CSV download

## Machine Learning Results

The current Isolation Forest configuration uses a contamination setting of 10%.

| Result | Observations |
|---|---:|
| Total valid observations | 399 |
| Normal observations | 359 |
| Unusual observations | 40 |
| Unusual observation percentage | 10% |

### Sensitivity Analysis

The model was also tested with different contamination settings.

| Contamination | Unusual observations |
|---|---:|
| 5% | 20 |
| 10% | 40 |
| 15% | 60 |

Twenty observations were consistently identified as unusual across all three tested settings.

**Important:** Contamination is a model configuration that controls the expected proportion of anomalies. It does not represent the actual percentage of drought-affected agricultural areas. The model has not been validated against independently labelled drought events, so accuracy, precision, recall, and drought-prediction performance are not claimed.

## Project Structure

```text
Crop Health Drought Monitoring/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── monthly/
│
├── notebooks/
│
├── src/
│   ├── gee_config.py
│   ├── check_india_datasets.py
│   ├── generate_india_locations.py
│   ├── download_india_satellite.py
│   ├── download_india_weather.py
│   ├── merge_india_satellite_weather.py
│   ├── check_india_dataset.py
│   ├── create_india_drought_features.py
│   ├── train_india_anomaly_model.py
│   └── test_india_model_sensitivity.py
│
├── models/
│   ├── india_isolation_forest.pkl
│   └── india_scaler.pkl
│
├── outputs/
│
├── dashboard/
│   ├── app.py
│   └── app_kadapa_backup.py
│
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/praveena917/Crop-Health-Drought-Monitoring.git
cd Crop-Health-Drought-Monitoring
```

Replace the repository URL with the actual GitHub repository URL if you use a different repository name.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Google Earth Engine

- Create or select a Google Cloud project.
- Enable the Earth Engine API.
- Complete Earth Engine authentication.
- Configure the project ID in `src/gee_config.py`.

Authenticate:

```bash
python -c "import ee; ee.Authenticate()"
```

Initialize Earth Engine:

```bash
python src/gee_config.py
```

### 5. Run the dashboard

From the project root:

```bash
python -m streamlit run dashboard/app.py
```

Open the local dashboard at:

```text
http://localhost:8501
```

**Note:** Running the dashboard requires the processed datasets and saved model outputs to be present in their expected project directories. Reproducing the complete data pipeline also requires Earth Engine authentication and access to the NASA POWER API.


### Future Improvements

- Expand the spatial sample using state-stratified and district-level sampling.
- Improve satellite-data coverage and quality checks.
- Integrate official drought and agricultural ground-truth datasets.
- Develop validated drought classification and severity models.
- Explore temporal models for vegetation and rainfall forecasting.
- Add crop-specific monitoring and regional comparisons.
- Deploy the dashboard as a cloud-hosted application.

## Author

**Praveena Kamanuru**

---
