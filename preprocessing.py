import logging
import numpy as np
import pandas as pd
from config import (
    PROCESSED_DATA_DIR,
    TARGET_DATETIME_FORMAT,
    DELAY_THRESHOLD_MINUTES,
    OUTLIER_MULTIPLIER,
)


def ensure_processed_path():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def standardize_airport_codes(df: pd.DataFrame, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip()
    return df


def parse_datetime_columns(df: pd.DataFrame, datetime_columns):
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def remove_duplicates_and_cancelled_flights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    if "Cancelled" in df.columns:
        df = df[df["Cancelled"] != True].copy()
    return df


def detect_and_cap_outliers(df: pd.DataFrame, numeric_cols):
    outlier_bounds = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - OUTLIER_MULTIPLIER * iqr
        upper = q3 + OUTLIER_MULTIPLIER * iqr
        outlier_bounds[col] = (lower, upper)
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
        logging.info("Outlier bounds for %s: lower=%.2f upper=%.2f", col, lower, upper)
    return df, outlier_bounds


def build_kpis(flight_ops: pd.DataFrame, fuel: pd.DataFrame, traffic: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    logging.info("Building KPI enrichments for flight operations.")
    flight_ops = flight_ops.copy()
    flight_ops["TotalDelayMinutes"] = flight_ops["DelayMinutes"].fillna(0).astype(float)
    flight_ops["DelayCategory"] = pd.cut(
        flight_ops["TotalDelayMinutes"],
        bins=[-1, 0, 15, 45, 120, np.inf],
        labels=["On-Time", "Minor", "Moderate", "Severe", "Extreme"],
    )
    flight_ops["DelayCostIndex"] = flight_ops["TotalDelayMinutes"] * 1.0
    flight_ops["DelayCost"] = flight_ops["TotalDelayMinutes"] * 100.0
    flight_ops["RouteEfficiencyScore"] = (
        100 - (flight_ops["TotalDelayMinutes"] / (flight_ops["FlightDurationMinutes"] + 1) * 12)
    ).clip(lower=20, upper=100)
    flight_ops["OnTimePerformancePct"] = np.where(
        flight_ops["TotalDelayMinutes"] <= DELAY_THRESHOLD_MINUTES,
        100.0,
        0.0,
    )
    flight_ops["FuelCostPerRoute"] = flight_ops["FuelUsedGallons"] * fuel["FuelPrice"].mean()
    fuel_volatility = fuel["FuelPrice"].pct_change().fillna(0).abs().rolling(7, min_periods=1).mean()
    flight_ops["FuelPriceVolatility"] = np.interp(
        flight_ops.index,
        range(len(fuel_volatility)),
        fuel_volatility.fillna(0).values,
    )
    traffic_summary = (
        traffic.groupby(["AirportCode", "Date"]).agg({"TrafficVolume": ["mean", "max"]}).reset_index()
    )
    traffic_summary.columns = ["AirportCode", "Date", "AvgTrafficVolume", "PeakTrafficLoad"]
    flight_ops = flight_ops.merge(
        traffic_summary,
        how="left",
        left_on=["Origin", "Date"],
        right_on=["AirportCode", "Date"],
    )
    flight_ops["CongestionSeverityIndex"] = np.interp(
        flight_ops["AvgTrafficVolume"].fillna(0),
        [flight_ops["AvgTrafficVolume"].min(), flight_ops["AvgTrafficVolume"].max()],
        [0, 100],
    )
    review_summary = (
        sentiment.groupby(["AirportCode", "ReviewDate"])
        .agg({"SentimentScore": "mean"})
        .reset_index()
        .rename(columns={"ReviewDate": "Date", "SentimentScore": "AvgSentimentScore"})
    )
    flight_ops = flight_ops.merge(
        review_summary,
        how="left",
        left_on=["Origin", "Date"],
        right_on=["AirportCode", "Date"],
    )
    flight_ops["SentimentScore"] = flight_ops["AvgSentimentScore"].fillna(0.0)
    flight_ops["SentimentCategory"] = pd.cut(
        flight_ops["SentimentScore"],
        bins=[-1.0, -0.05, 0.05, 1.0],
        labels=["Negative", "Neutral", "Positive"],
    ).astype(str)
    flight_ops["CustomerSatisfactionIntensity"] = (flight_ops["SentimentScore"].abs() * 100).fillna(0).round(1)
    return flight_ops


def prepare_datasets(raw_data: dict) -> pd.DataFrame:
    ensure_processed_path()
    flight_ops = raw_data["flight_ops"]
    weather = raw_data["weather"]
    traffic = raw_data["traffic"]
    fuel = raw_data["fuel"]
    reviews = raw_data["reviews"]

    flight_ops = standardize_airport_codes(flight_ops, ["Origin", "Destination"])
    weather = standardize_airport_codes(weather, ["AirportCode"])
    traffic = standardize_airport_codes(traffic, ["AirportCode"])
    reviews = standardize_airport_codes(reviews, ["AirportCode"])

    flight_ops = parse_datetime_columns(
        flight_ops,
        ["ScheduledDeparture", "ActualDeparture", "ScheduledArrival", "ActualArrival"],
    )
    reviews["ReviewDate"] = pd.to_datetime(reviews["ReviewDate"], errors="coerce").dt.strftime("%Y-%m-%d")

    flight_ops = remove_duplicates_and_cancelled_flights(flight_ops)
    numeric_outliers = ["DelayMinutes", "FuelUsedGallons", "FlightDurationMinutes"]
    flight_ops, bounds = detect_and_cap_outliers(flight_ops, numeric_outliers)

    flight_ops["Date"] = flight_ops["ScheduledDeparture"].dt.strftime("%Y-%m-%d")
    merged = build_kpis(flight_ops, fuel, traffic, reviews)

    sentiment_file = PROCESSED_DATA_DIR / "processed_sentiment.csv"
    merged.to_csv(PROCESSED_DATA_DIR / "processed_flights.csv", index=False)
    reviews.to_csv(sentiment_file, index=False)
    logging.info("Processed dataset persisted: %s", PROCESSED_DATA_DIR / "processed_flights.csv")
    logging.info("Processed sentiment persisted: %s", sentiment_file)
    return merged
