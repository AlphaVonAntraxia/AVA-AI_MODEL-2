from pathlib import Path

import joblib
import pandas as pd

from models.train_model import build_features, FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"


def predict_next_return(ticker: str) -> float:
    data_path = DATA_DIR / f"{ticker}_daily.csv"
    model_path = MODEL_DIR / f"{ticker}_random_forest.joblib"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    data = build_features(data_path)

    latest = (
        data[FEATURE_COLUMNS]
        .dropna()
        .iloc[[-1]]
    )

    model = joblib.load(model_path)

    prediction = model.predict(latest)[0]

    return float(prediction)


if __name__ == "__main__":
    ticker = "AAPL"

    prediction = predict_next_return(ticker)

    print(f"{ticker} predicted next-day return: {prediction:.6%}")