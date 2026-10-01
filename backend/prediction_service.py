from models.availability import get_data_freshness
from models.predict import predict_next_return


MODEL_VERSION = "v2"
MODEL_TYPE = "random_forest"
PREDICTION_HORIZON = "next_trading_day"


def get_prediction(ticker: str) -> dict:
    ticker = ticker.upper()

    prediction_data = predict_next_return(
    ticker,
    return_metadata=True,
    )

    return {
    "ticker": ticker,
    "predicted_next_day_return": prediction_data["prediction"],
    "latest_data_date": prediction_data["latest_data_date"],
    "freshness": get_data_freshness(
        prediction_data["latest_data_date"]
    ),
    "model_version": MODEL_VERSION,
    "model_type": MODEL_TYPE,
    "prediction_horizon": PREDICTION_HORIZON,
    }