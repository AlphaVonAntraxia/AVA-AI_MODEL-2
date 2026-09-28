from models.predict import predict_next_return


def get_prediction(ticker: str) -> dict:
    ticker = ticker.upper()

    prediction = predict_next_return(ticker)

    return {
        "ticker": ticker,
        "predicted_next_day_return": prediction,
    }