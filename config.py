from pathlib import Path

# Root project paths
ROOT_DIR = Path(__file__).parent.resolve()
RAW_DATA_DIR = ROOT_DIR / "raw_data"
PROCESSED_DATA_DIR = ROOT_DIR / "processed_data"
EXPORT_DIR = ROOT_DIR / "exports"
REPORT_DIR = ROOT_DIR / "reports"
MODEL_DIR = ROOT_DIR / "models"

# Raw filenames
FLIGHT_OPS_FILE = RAW_DATA_DIR / "flight_operations.csv"
WEATHER_FILE = RAW_DATA_DIR / "weather.csv"
TRAFFIC_FILE = RAW_DATA_DIR / "airport_traffic.csv"
FUEL_FILE = RAW_DATA_DIR / "fuel_prices.csv"
REVIEWS_FILE = RAW_DATA_DIR / "passenger_reviews.csv"

# Export filenames
FACT_FLIGHTS_FILE = EXPORT_DIR / "Fact_Flights.csv"
FACT_DELAYS_FILE = EXPORT_DIR / "Fact_Delays.csv"
DIM_DATE_FILE = EXPORT_DIR / "Dim_Date.csv"
DIM_ROUTE_FILE = EXPORT_DIR / "Dim_Route.csv"
DIM_AIRPORT_FILE = EXPORT_DIR / "Dim_Airport.csv"
DIM_WEATHER_FILE = EXPORT_DIR / "Dim_Weather.csv"
DIM_FUEL_FILE = EXPORT_DIR / "Dim_Fuel.csv"
DIM_SENTIMENT_FILE = EXPORT_DIR / "Dim_Sentiment.csv"
REPORT_FILE = REPORT_DIR / "Executive_Report.txt"

# Data processing constants
DELAY_THRESHOLD_MINUTES = 15
OUTLIER_MULTIPLIER = 1.5
RANDOM_SEED = 42

# ML configuration
CLASSIFICATION_MODELS = ["random_forest"]
REGRESSION_MODELS = ["gradient_boosting"]

# Date and time formats
TARGET_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"

# Airline KPI weights
DELAY_COST_PER_MINUTE = 100.0
CREW_EXTENSION_COST_PER_HOUR = 250.0
AIRCRAFT_IDLE_COST_PER_HOUR = 800.0
FUEL_WASTAGE_PERCENTAGE = 0.08
