import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    confusion_matrix,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
from config import MODEL_DIR, DELAY_THRESHOLD_MINUTES


def ensure_model_path():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def prepare_ml_features(flights: pd.DataFrame) -> pd.DataFrame:
    features = flights.copy()
    features["DepartureHour"] = features["ScheduledDeparture"].dt.hour
    features["DayOfWeek"] = features["ScheduledDeparture"].dt.dayofweek
    features["WeatherSeverity"] = np.clip((10 - features["AvgTrafficVolume"].fillna(0)) * 5, 0, 100)
    features["HistoricalRouteDelay"] = features.groupby(["Origin", "Destination"])["TotalDelayMinutes"].transform("mean")
    selected = features[
        [
            "WeatherSeverity",
            "CongestionSeverityIndex",
            "HistoricalRouteDelay",
            "FuelPriceVolatility",
            "DepartureHour",
            "DayOfWeek",
            "RouteEfficiencyScore",
            "SentimentScore",
            "TotalDelayMinutes",
        ]
    ].copy()
    return selected


def build_classification_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def build_regression_model(X_train, y_train):
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "rmse": mean_squared_error(y_test, y_pred, squared=False),
        "mae": mean_absolute_error(y_test, y_pred),
    }


def train_predict_models(flights: pd.DataFrame) -> dict:
    ensure_model_path()
    dataset = prepare_ml_features(flights)
    dataset = dataset.fillna(dataset.median())
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(dataset.drop(columns=["TotalDelayMinutes"]))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_reg = dataset["TotalDelayMinutes"].astype(float).values
    y_cls = (y_reg >= DELAY_THRESHOLD_MINUTES).astype(int)

    X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
        X_scaled, y_cls, y_reg, test_size=0.25, random_state=42, stratify=y_cls
    )

    clf = build_classification_model(X_train, y_train_cls)
    reg = build_regression_model(X_train, y_train_reg)

    classification_metrics = evaluate_classification(clf, X_test, y_test_cls)
    regression_metrics = evaluate_regression(reg, X_test, y_test_reg)

    joblib.dump(clf, MODEL_DIR / "delay_classification_model.joblib")
    joblib.dump(reg, MODEL_DIR / "delay_regression_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "feature_scaler.joblib")
    joblib.dump(imputer, MODEL_DIR / "feature_imputer.joblib")

    feature_importances = {
        "classification": dict(zip(
            [
                "WeatherSeverity",
                "CongestionSeverityIndex",
                "HistoricalRouteDelay",
                "FuelPriceVolatility",
                "DepartureHour",
                "DayOfWeek",
                "RouteEfficiencyScore",
                "SentimentScore",
            ],
            clf.feature_importances_.tolist(),
        )),
        "regression": dict(zip(
            [
                "WeatherSeverity",
                "CongestionSeverityIndex",
                "HistoricalRouteDelay",
                "FuelPriceVolatility",
                "DepartureHour",
                "DayOfWeek",
                "RouteEfficiencyScore",
                "SentimentScore",
            ],
            reg.feature_importances_.tolist(),
        )),
    }

    predictions = {
        "classification": clf.predict(X_scaled).astype(int),
        "regression": reg.predict(X_scaled).round(1),
    }
    flights = flights.copy()
    flights["PredictedDelayMinutes"] = predictions["regression"]
    flights["PredictedDelayFlag"] = np.where(predictions["classification"] == 1, "Delayed", "On-Time")
    flights["ActualDelayFlag"] = np.where(flights["TotalDelayMinutes"] >= DELAY_THRESHOLD_MINUTES, "Delayed", "On-Time")
    flights["PredictedErrorMinutes"] = (flights["PredictedDelayMinutes"] - flights["TotalDelayMinutes"]).abs()

    logging.info("Model training completed and artifacts saved.")
    return {
        "metrics": {
            "classification": classification_metrics,
            "regression": regression_metrics,
        },
        "feature_importances": feature_importances,
        "predictions": flights,
    }
