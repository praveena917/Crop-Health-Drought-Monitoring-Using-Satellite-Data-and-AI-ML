
import ee

PROJECT_ID = "crop-health-monitoring-514608"

def initialize_gee():
    ee.Initialize(project=PROJECT_ID)
    print("Google Earth Engine initialized successfully!")

if __name__ == "__main__":
    initialize_gee()
