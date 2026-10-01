from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"


def get_data_freshness(latest_data_date: str) -> dict:
    latest_date = datetime.strptime(
        latest_data_date,
        "%Y-%m-%d",
    ).date()

    current_date = datetime.now().date()

    age_days = (
        current_date - latest_date
    ).days

    return {
        "latest_data_date": latest_data_date,
        "data_age_days": age_days,
        "is_stale": age_days > 1,
    }


def get_prediction_availability(
    ticker: str,
) -> dict:
    ticker = ticker.upper()

    data_exists = (
        DATA_DIR / f"{ticker}_daily.csv"
    ).exists()

    model_exists = (
        MODEL_DIR / f"{ticker}_random_forest.joblib"
    ).exists()

    latest_data_date = None

    if data_exists:
        data = pd.read_csv(
            DATA_DIR / f"{ticker}_daily.csv",
            parse_dates=["datetime"],
        )

        latest_data_date = (
            data["datetime"]
            .max()
            .strftime("%Y-%m-%d")
        )

    freshness = (
        get_data_freshness(latest_data_date)
        if latest_data_date
        else None
    )

    return {
        "ticker": ticker,
        "data_available": data_exists,
        "model_available": model_exists,
        "prediction_available": (
            data_exists and model_exists
        ),
        "latest_data_date": latest_data_date,
        "freshness": freshness,
    }


if __name__ == "__main__":
    print(
        get_prediction_availability("AAPL")
    )