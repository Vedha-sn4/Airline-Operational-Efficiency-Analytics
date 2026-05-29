import logging
import numpy as np
import pandas as pd
from config import DELAY_THRESHOLD_MINUTES


def route_efficiency_analysis(flights: pd.DataFrame) -> pd.DataFrame:
    logging.info("Running route efficiency analysis.")
    route_perf = (
        flights.groupby(["Origin", "Destination", "RouteDistance"])
        .agg(
            flights_per_route=("FlightID", "count"),
            avg_delay=("TotalDelayMinutes", "mean"),
            delay_frequency=("DelayCategory", lambda x: (x != "On-Time").mean()),
            avg_efficiency=("RouteEfficiencyScore", "mean"),
            avg_fuel_cost=("FuelCostPerRoute", "mean"),
        )
        .reset_index()
    )
    route_perf["InefficiencyIndex"] = (
        route_perf["avg_delay"] * 0.5 + route_perf["delay_frequency"] * 50 + (100 - route_perf["avg_efficiency"]) * 0.5
    )
    logging.info("Route efficiency analysis completed.")
    return route_perf.sort_values(by="InefficiencyIndex", ascending=False)


def airport_congestion_analysis(flights: pd.DataFrame, weather: pd.DataFrame, traffic: pd.DataFrame) -> pd.DataFrame:
    logging.info("Running airport congestion analysis.")
    weather_summary = weather.groupby(["AirportCode", "Date"]).agg(
        weather_severity=("VisibilityMiles", lambda x: (10 - x.mean()) * 10),
    )
    traffic_summary = traffic.groupby(["AirportCode", "Date"]).agg(
        traffic_load=("TrafficVolume", "mean"),
    )
    delays = flights.groupby(["Origin", "Date"]).agg(
        avg_dep_delay=("TotalDelayMinutes", "mean"),
        avg_arr_delay=("TotalDelayMinutes", "mean"),
    )
    delays = delays.reset_index().rename(columns={"Origin": "AirportCode"})
    joined = (
        delays.merge(weather_summary.reset_index(), on=["AirportCode", "Date"], how="left")
        .merge(traffic_summary.reset_index(), on=["AirportCode", "Date"], how="left")
    )
    joined["OperationalRiskScore"] = (
        joined["avg_dep_delay"].fillna(0) * 0.6
        + joined["weather_severity"].fillna(0) * 0.25
        + joined["traffic_load"].fillna(0) * 0.15
    )
    logging.info("Airport congestion analysis completed.")
    return joined.sort_values(by="OperationalRiskScore", ascending=False)


def financial_impact_attribution(flights: pd.DataFrame) -> pd.DataFrame:
    logging.info("Estimating financial impact attribution.")
    flights = flights.copy()
    flights["DelayCost"] = flights["TotalDelayMinutes"] * 100.0
    flights["IdleAircraftCost"] = (flights["FlightDurationMinutes"] / 60).clip(lower=0) * 800.0
    flights["CrewExtensionCost"] = np.where(
        flights["TotalDelayMinutes"] > DELAY_THRESHOLD_MINUTES,
        ((flights["TotalDelayMinutes"] - DELAY_THRESHOLD_MINUTES) / 60) * 250.0,
        0.0,
    )
    flights["FuelWastageCost"] = flights["FuelCostPerRoute"] * 0.08
    agg = (
        flights.groupby(["Airline", "Origin", "Destination", "Date"])
        .agg(
            total_delay_cost=("DelayCost", "sum"),
            total_idle_cost=("IdleAircraftCost", "sum"),
            total_crew_cost=("CrewExtensionCost", "sum"),
            total_fuel_wastage=("FuelWastageCost", "sum"),
            avg_route_efficiency=("RouteEfficiencyScore", "mean"),
        )
        .reset_index()
    )
    logging.info("Financial impact attribution completed.")
    return agg
