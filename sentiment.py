import logging
from pathlib import Path
import pandas as pd

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None


def analyze_sentiment(reviews: pd.DataFrame) -> pd.DataFrame:
    """Apply sentiment analysis to passenger review text."""
    if SentimentIntensityAnalyzer is None:
        raise ImportError(
            "vaderSentiment is required for sentiment analysis. Install with `pip install vaderSentiment`."
        )

    analyzer = SentimentIntensityAnalyzer()
    sentiment_records = []
    for _, row in reviews.iterrows():
        text = str(row.get("ReviewText", ""))
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            category = "Positive"
        elif compound <= -0.05:
            category = "Negative"
        else:
            category = "Neutral"

        sentiment_records.append(
            {
                "ReviewID": row.get("ReviewID"),
                "ReviewDate": row.get("ReviewDate"),
                "AirportCode": row.get("AirportCode"),
                "ReviewText": text,
                "SentimentScore": round(compound, 4),
                "SentimentCategory": category,
                "CustomerSatisfactionIntensity": round(abs(compound) * 100, 1),
            }
        )

    logging.info("Completed sentiment scoring for %d reviews.", len(sentiment_records))
    return pd.DataFrame(sentiment_records)
