from pathlib import Path

import joblib

from models.train_model import build_features, FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"


def predict_next_return(
    ticker: str,
    return_metadata: bool = False,
):
    ticker = ticker.upper()

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

    latest_raw_date = data.index.max()

    latest = (
        data[FEATURE_COLUMNS]
        .dropna()
        .iloc[[-1]]
    )

    model = joblib.load(model_path)

    prediction = float(
        model.predict(latest)[0]
    )

    if return_metadata:
        return {
            "prediction": prediction,
            "latest_data_date": latest_raw_date.strftime(
                "%Y-%m-%d"
            ),
        }

    return prediction


if __name__ == "__main__":
    ticker = "AAPL"

    prediction = predict_next_return(ticker)

    print(
        f"{ticker} predicted next-day return: "
        f"{prediction:.6%}"
    )