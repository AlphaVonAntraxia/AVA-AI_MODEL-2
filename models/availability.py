from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"


def get_prediction_availability(ticker: str) -> dict:
    ticker = ticker.upper()

    data_exists = (DATA_DIR / f"{ticker}_daily.csv").exists()
    model_exists = (MODEL_DIR / f"{ticker}_random_forest.joblib").exists()

    return {
        "ticker": ticker,
        "data_available": data_exists,
        "model_available": model_exists,
        "prediction_available": data_exists and model_exists,
    }
