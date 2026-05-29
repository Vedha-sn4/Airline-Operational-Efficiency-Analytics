import logging
import pandas as pd
from pathlib import Path
from config import (
    EXPORT_DIR,
    FACT_FLIGHTS_FILE,
    FACT_DELAYS_FILE,
    DIM_DATE_FILE,
    DIM_ROUTE_FILE,
    DIM_AIRPORT_FILE,
    DIM_WEATHER_FILE,
    DIM_FUEL_FILE,
    DIM_SENTIMENT_FILE,
)


def ensure_export_path():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def build_dimensions(flights: pd.DataFrame, weather: pd.DataFrame, fuel: pd.DataFrame, sentiment: pd.DataFrame):
    ensure_export_path()
    date_dim = (
        flights["Date"].drop_duplicates()
        .sort_values()
        .to_frame(name="Date")
        .reset_index(drop=True)
    )
    date_dim["DateID"] = date_dim.index + 1
    date_dim["Month"] = pd.to_datetime(date_dim["Date"]).dt.month
    date_dim["Quarter"] = pd.to_datetime(date_dim["Date"]).dt.quarter
    date_dim["Year"] = pd.to_datetime(date_dim["Date"]).dt.year
    date_dim["DayOfWeek"] = pd.to_datetime(date_dim["Date"]).dt.day_name()

    route_dim = (
        flights[["Origin", "Destination", "RouteDistance"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    route_dim["RouteID"] = route_dim.index + 1

    airport_dim = (
        flights[["Origin", "Destination"]]
        .melt(value_name="AirportCode")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    airport_dim["AirportID"] = airport_dim.index + 1
    airport_dim["AirportName"] = airport_dim["AirportCode"]
    airport_dim["City"] = airport_dim["AirportCode"].map(
        {"JFK": "New York", "LAX": "Los Angeles", "ORD": "Chicago", "ATL": "Atlanta", "DFW": "Dallas"}
    )
    airport_dim["Country"] = "USA"
    airport_dim["CongestionSeverity"] = flights.groupby("Origin")["CongestionSeverityIndex"].mean().reindex(airport_dim["AirportCode"]).values

    weather_dim = (
        weather[["Date", "AirportCode", "WeatherCondition", "Temperature", "VisibilityMiles", "WindSpeedMph"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    weather_dim["WeatherID"] = weather_dim.index + 1
    weather_dim["WeatherSeverity"] = ((10 - weather_dim["VisibilityMiles"]).clip(lower=0) * 10).round(1)

    fuel_dim = fuel.drop_duplicates().reset_index(drop=True)
    fuel_dim["FuelID"] = fuel_dim.index + 1
    fuel_dim["FuelVolatilityIndex"] = fuel_dim["FuelPrice"].pct_change().fillna(0).abs().rolling(7, min_periods=1).mean().round(4)

    sentiment_dim = sentiment[["ReviewID", "ReviewText", "SentimentScore", "SentimentCategory"]].drop_duplicates().reset_index(drop=True)
    sentiment_dim["SentimentID"] = sentiment_dim.index + 1

    return {
        "date_dim": date_dim,
        "route_dim": route_dim,
        "airport_dim": airport_dim,
        "weather_dim": weather_dim,
        "fuel_dim": fuel_dim,
        "sentiment_dim": sentiment_dim,
    }


def build_facts(flights: pd.DataFrame, dims: dict):
    date_dim = dims["date_dim"]
    route_dim = dims["route_dim"]
    airport_dim = dims["airport_dim"]
    weather_dim = dims["weather_dim"]
    fuel_dim = dims["fuel_dim"]
    sentiment_dim = dims["sentiment_dim"]

    fact_flights = flights.copy()
    fact_flights = fact_flights.merge(date_dim[["DateID", "Date"]], on="Date", how="left")
    fact_flights = fact_flights.merge(
        route_dim,
        left_on=["Origin", "Destination", "RouteDistance"],
        right_on=["Origin", "Destination", "RouteDistance"],
        how="left",
    )
    fact_flights = fact_flights.merge(
        airport_dim[["AirportID", "AirportCode"]],
        left_on="Origin",
        right_on="AirportCode",
        how="left",
    )
    fact_flights = fact_flights.drop(columns=["AirportCode"], errors="ignore")
    fact_flights = fact_flights.merge(
        weather_dim[["Date", "AirportCode", "WeatherID"]],
        left_on=["Date", "Origin"],
        right_on=["Date", "AirportCode"],
        how="left",
    )
    fact_flights = fact_flights.merge(
        fuel_dim[["Date", "FuelID"]],
        left_on="Date",
        right_on="Date",
        how="left",
    )
    fact_flights = fact_flights[
        [
            "FlightID",
            "DateID",
            "RouteID",
            "AirportID",
            "WeatherID",
            "FuelID",
            "DelayMinutes",
            "DelayCost",
            "RouteEfficiencyScore",
            "SentimentScore",
        ]
    ]

    fact_delays = flights.copy()
    fact_delays["DelayID"] = range(1, len(fact_delays) + 1)
    fact_delays = fact_delays.merge(
        fact_flights[["FlightID", "DateID"]], on="FlightID", how="left"
    )
    fact_delays = fact_delays[
        [
            "DelayID",
            "FlightID",
            "DelayCategory",
            "TotalDelayMinutes",
            "PredictedDelayMinutes",
            "DelayMinutes",
        ]
    ].rename(columns={
        "TotalDelayMinutes": "DelayDuration",
        "PredictedDelayMinutes": "PredictedDelay",
        "DelayMinutes": "ActualDelay",
    })

    fact_flights.to_csv(FACT_FLIGHTS_FILE, index=False)
    fact_delays.to_csv(FACT_DELAYS_FILE, index=False)
    date_dim.to_csv(DIM_DATE_FILE, index=False)
    route_dim.to_csv(DIM_ROUTE_FILE, index=False)
    airport_dim.to_csv(DIM_AIRPORT_FILE, index=False)
    weather_dim.to_csv(DIM_WEATHER_FILE, index=False)
    fuel_dim.to_csv(DIM_FUEL_FILE, index=False)
    sentiment_dim.to_csv(DIM_SENTIMENT_FILE, index=False)

    logging.info("Exported star schema CSV files to %s", EXPORT_DIR)
    return fact_flights, fact_delays
