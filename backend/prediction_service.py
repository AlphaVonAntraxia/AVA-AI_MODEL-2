from models.predict import predict_next_return


MODEL_VERSION = "v2"
MODEL_TYPE = "random_forest"
PREDICTION_HORIZON = "next_trading_day"


def get_prediction(ticker: str) -> dict:
    ticker = ticker.upper()

    prediction = predict_next_return(ticker)

    return {
        "ticker": ticker,
        "predicted_next_day_return": prediction,
        "model_version": MODEL_VERSION,
        "model_type": MODEL_TYPE,
        "prediction_horizon": PREDICTION_HORIZON,
    }