from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.prediction_service import get_prediction


SUPPORTED_TICKERS = {
    "AAPL",
    "AMZN",
    "GOOGL",
    "MSFT",
    "NVDA",
    "TSLA",
}


class PredictionResponse(BaseModel):
    ticker: str
    predicted_next_day_return: float


app = FastAPI(
    title="AVA-AI API",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "AVA-AI API is running"
    }


@app.get(
    "/predict/{ticker}",
    response_model=PredictionResponse,
)
def predict(ticker: str):
    ticker = ticker.upper()

    if ticker not in SUPPORTED_TICKERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported ticker '{ticker}'. "
                f"Supported tickers: {', '.join(sorted(SUPPORTED_TICKERS))}"
            ),
        )

    return get_prediction(ticker)