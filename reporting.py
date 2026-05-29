import logging
from pathlib import Path
from config import REPORT_DIR, REPORT_FILE


def ensure_report_path():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def generate_executive_report(
    preprocessing_summary: dict,
    route_insights: dict,
    congestion_insights: dict,
    financial_metrics: dict,
    model_metrics: dict,
    recommendations: list,
):
    ensure_report_path()
    report_lines = [
        "Multi-Domain Airline Optimization Pipeline Executive Summary",
        "============================================================",
        "\nPIPELINE OVERVIEW",
        "The analytics pipeline integrates flight operations, weather, airport traffic, fuel pricing, and passenger sentiment domains to produce clean, relational datasets optimized for Power BI ingestion.",
        "\nPREPROCESSING SUMMARY",
        f"Raw datasets were standardized and merged across five domains. Missing values and duplicates were handled, cancelled flights were removed, and numeric outliers were capped using the IQR method.",
        f"A total of {preprocessing_summary.get('flight_records', 'N/A')} flight records were processed.",
        f"Sentiment extraction produced {preprocessing_summary.get('review_records', 'N/A')} scored passenger reviews.",
        "\nTOP OPERATIONAL INSIGHTS",
        f"High-inefficiency route: {route_insights.get('top_route', 'Unknown')} with inefficiency index {route_insights.get('top_value', 'N/A'):.2f}.",
        f"Bottleneck airport: {congestion_insights.get('top_airport', 'Unknown')} with operational risk score {congestion_insights.get('top_value', 'N/A'):.2f}.",
        "\nDELAY RISK FINDINGS",
        f"Average delay cost per route segment is estimated at ${financial_metrics.get('avg_delay_cost', 0):.2f}.",
        f"Crew extension and idle aircraft cost drivers were identified across routes with average route efficiency below 70%.",
        "\nMODEL PERFORMANCE",
        f"Delay classification accuracy: {model_metrics['classification'].get('accuracy', 0):.2f}.",
        f"Precision: {model_metrics['classification'].get('precision', 0):.2f}, Recall: {model_metrics['classification'].get('recall', 0):.2f}, F1-score: {model_metrics['classification'].get('f1_score', 0):.2f}.",
        f"Regression RMSE: {model_metrics['regression'].get('rmse', 0):.2f}, MAE: {model_metrics['regression'].get('mae', 0):.2f}.",
        "\nRECOMMENDATIONS",
    ]
    report_lines.extend([f"- {item}" for item in recommendations])
    report_lines.extend(
        [
            "\nSTAR SCHEMA GUIDANCE",
            "The exported schema separates facts and dimensions to simplify Power BI modelling. Fact_Flights links to Date, Route, Airport, Weather, Fuel, and Sentiment dimensions. Fact_Delays isolates delay-specific performance metrics.",
            "Date keys should be used for time intelligence, Route keys for operational analysis, and Airport keys for congestion and hub performance dashboards.",
        ]
    )

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logging.info("Executive report generated at %s", REPORT_FILE)
    return REPORT_FILE
