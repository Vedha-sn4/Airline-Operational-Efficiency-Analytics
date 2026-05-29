import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import (
    RAW_DATA_DIR,
    FLIGHT_OPS_FILE,
    WEATHER_FILE,
    TRAFFIC_FILE,
    FUEL_FILE,
    REVIEWS_FILE,
    RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def ensure_raw_data_path():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_raw_data() -> None:
    """Generate sample raw datasets for the airline analytics pipeline."""
    logging.info("Generating synthetic raw datasets in %s", RAW_DATA_DIR)
    ensure_raw_data_path()

    np.random.seed(RANDOM_SEED)
    airlines = ["AvaAir", "BlueJet", "CloudFlight", "DeltaSky", "EliteWings"]
    airports = [
        {"code": "JFK", "name": "John F Kennedy Intl", "city": "New York", "country": "USA"},
        {"code": "LAX", "name": "Los Angeles Intl", "city": "Los Angeles", "country": "USA"},
        {"code": "ORD", "name": "O'Hare Intl", "city": "Chicago", "country": "USA"},
        {"code": "ATL", "name": "Hartsfield-Jackson", "city": "Atlanta", "country": "USA"},
        {"code": "DFW", "name": "Dallas/Fort Worth", "city": "Dallas", "country": "USA"},
    ]
    routes = [
        ("JFK", "LAX", 2475),
        ("JFK", "ORD", 740),
        ("ATL", "DFW", 732),
        ("LAX", "ORD", 1744),
        ("ORD", "ATL", 606),
    ]

    start_date = datetime.today().replace(hour=5, minute=0, second=0, microsecond=0)
    flight_rows = []
    weather_rows = []
    traffic_rows = []
    fuel_rows = []
    review_rows = []

    for day_offset in range(30):
        current_date = start_date + timedelta(days=day_offset)
        for route_id, (origin, destination, distance) in enumerate(routes, start=1):
            for flight_seq in range(1, 4):
                scheduled_dep = current_date + timedelta(hours=flight_seq * 3)
                delay = int(np.random.normal(loc=12, scale=18))
                delay = max(-10, delay)
                cancelled = np.random.rand() < 0.06
                departure_time = scheduled_dep + timedelta(minutes=delay if not cancelled else 0)
                arrival_time = departure_time + timedelta(minutes=int(distance / 8.5) + np.random.randint(-15, 30))
                flight_rows.append(
                    {
                        "FlightID": f"FL{day_offset:02d}{route_id}{flight_seq}",
                        "Airline": np.random.choice(airlines),
                        "Origin": origin,
                        "Destination": destination,
                        "RouteDistance": distance,
                        "ScheduledDeparture": scheduled_dep.strftime("%Y-%m-%d %H:%M:%S"),
                        "ActualDeparture": departure_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "ScheduledArrival": (scheduled_dep + timedelta(minutes=int(distance / 8.5))).strftime("%Y-%m-%d %H:%M:%S"),
                        "ActualArrival": arrival_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "DelayMinutes": max(0, delay),
                        "Cancelled": cancelled,
                        "FuelUsedGallons": round(distance * np.random.uniform(4.8, 5.6), 1),
                        "FlightDurationMinutes": int((arrival_time - departure_time).total_seconds() / 60),
                        "Passengers": np.random.randint(80, 210),
                    }
                )

        # Weather and traffic records per date and airport
        for airport in airports:
            weather_rows.append(
                {
                    "Date": current_date.strftime("%Y-%m-%d"),
                    "AirportCode": airport["code"],
                    "WeatherCondition": np.random.choice(["Clear", "Rain", "Thunderstorm", "Fog", "Snow"] , p=[0.5, 0.25, 0.1, 0.1, 0.05]),
                    "Temperature": round(np.random.uniform(32, 90), 1),
                    "VisibilityMiles": round(np.random.uniform(1, 10), 1),
                    "WindSpeedMph": round(np.random.uniform(5, 28), 1),
                }
            )
            traffic_rows.append(
                {
                    "Date": current_date.strftime("%Y-%m-%d"),
                    "AirportCode": airport["code"],
                    "TrafficVolume": np.random.randint(300, 1200),
                    "PeakHour": np.random.choice(["08:00", "12:00", "17:00", "20:00"]),
                }
            )

        # Fuel price record for each day
        fuel_rows.append(
            {
                "Date": current_date.strftime("%Y-%m-%d"),
                "FuelPrice": round(np.random.uniform(2.10, 3.50), 3),
            }
        )

        # Passenger feedback samples each day
        for review_id in range(3):
            sentiment_texts = [
                "The flight was on time and staff were helpful.",
                "Delayed departure was frustrating, but the crew still tried.",
                "I loved the comfort but the baggage wait was too long.",
                "Crew service was poor and flight announcements were unclear.",
                "Smooth boarding while the weather made operations challenging.",
            ]
            review_rows.append(
                {
                    "ReviewID": f"RV{day_offset:02d}{review_id}",
                    "ReviewDate": current_date.strftime("%Y-%m-%d"),
                    "AirportCode": np.random.choice([a["code"] for a in airports]),
                    "ReviewText": np.random.choice(sentiment_texts),
                }
            )

    pd.DataFrame(flight_rows).to_csv(FLIGHT_OPS_FILE, index=False)
    pd.DataFrame(weather_rows).to_csv(WEATHER_FILE, index=False)
    pd.DataFrame(traffic_rows).to_csv(TRAFFIC_FILE, index=False)
    pd.DataFrame(fuel_rows).to_csv(FUEL_FILE, index=False)
    pd.DataFrame(review_rows).to_csv(REVIEWS_FILE, index=False)

    logging.info("Synthetic raw data generation completed.")


def load_raw_datasets():
    """Load raw CSV files into pandas DataFrames."""
    logging.info("Loading raw datasets from %s", RAW_DATA_DIR)
    try:
        flight_ops = pd.read_csv(FLIGHT_OPS_FILE)
        weather = pd.read_csv(WEATHER_FILE)
        traffic = pd.read_csv(TRAFFIC_FILE)
        fuel = pd.read_csv(FUEL_FILE)
        reviews = pd.read_csv(REVIEWS_FILE)
    except FileNotFoundError as exc:
        raise RuntimeError("Raw data files are missing. Run generate_synthetic_raw_data() first.") from exc

    return {
        "flight_ops": flight_ops,
        "weather": weather,
        "traffic": traffic,
        "fuel": fuel,
        "reviews": reviews,
    }
