import logging
from pathlib import Path
from data_loader import generate_synthetic_raw_data, load_raw_datasets
from preprocessing import prepare_datasets
from sentiment import analyze_sentiment
from analytics import route_efficiency_analysis, airport_congestion_analysis, financial_impact_attribution
from ml_models import train_predict_models
from export_schema import build_dimensions, build_facts
from reporting import generate_executive_report
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, EXPORT_DIR, REPORT_DIR, MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def ensure_project_dirs():
    for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXPORT_DIR, REPORT_DIR, MODEL_DIR]:
        Path(path).mkdir(parents=True, exist_ok=True)


def run_pipeline():
    logging.info("Starting Multi-Domain Airline Optimization Pipeline.")
    ensure_project_dirs()
    generate_synthetic_raw_data()
    raw_data = load_raw_datasets()

    sentiment_scores = analyze_sentiment(raw_data["reviews"])
    processed_flights = prepare_datasets({
        "flight_ops": raw_data["flight_ops"],
        "weather": raw_data["weather"],
        "traffic": raw_data["traffic"],
        "fuel": raw_data["fuel"],
        "reviews": sentiment_scores,
    })

    route_summary = route_efficiency_analysis(processed_flights)
    congestion_summary = airport_congestion_analysis(
        processed_flights,
        raw_data["weather"],
        raw_data["traffic"],
    )
    financial_summary = financial_impact_attribution(processed_flights)

    model_results = train_predict_models(processed_flights)
    predictions_df = model_results["predictions"]

    dims = build_dimensions(
        processed_flights,
        raw_data["weather"],
        raw_data["fuel"],
        sentiment_scores,
    )
    facts = build_facts(predictions_df, dims)

    report_file = generate_executive_report(
        preprocessing_summary={
            "flight_records": len(processed_flights),
            "review_records": len(sentiment_scores),
        },
        route_insights={
            "top_route": route_summary.iloc[0].to_dict() if not route_summary.empty else {},
            "top_value": route_summary.iloc[0].get("InefficiencyIndex", 0) if not route_summary.empty else 0,
        },
        congestion_insights={
            "top_airport": congestion_summary.iloc[0].get("AirportCode", "Unknown") if not congestion_summary.empty else "Unknown",
            "top_value": congestion_summary.iloc[0].get("OperationalRiskScore", 0) if not congestion_summary.empty else 0,
        },
        financial_metrics={
            "avg_delay_cost": financial_summary["total_delay_cost"].mean() if not financial_summary.empty else 0,
        },
        model_metrics=model_results["metrics"],
        recommendations=[
            "Prioritize flight schedule resilience on routes with the highest inefficiency indices.",
            "Align staffing and gate resources at high-risk airports during peak congestion windows.",
            "Monitor fuel volatility daily and hedge for routes with above-average fuel cost per route.",
            "Use sentiment trends to improve customer communication during delay events.",
        ],
    )

    logging.info("Pipeline completed successfully.")
    logging.info("Executive report written to %s", report_file)
    return {
        "processed_flights": processed_flights,
        "route_summary": route_summary,
        "congestion_summary": congestion_summary,
        "financial_summary": financial_summary,
        "model_results": model_results,
        "dimensions": dims,
        "report_file": report_file,
    }


if __name__ == "__main__":
    run_pipeline()
