from fastapi import FastAPI, HTTPException

from backend.prediction_service import get_prediction


SUPPORTED_TICKERS = {
    "AAPL",
    "AMZN",
    "GOOGL",
    "MSFT",
    "NVDA",
    "TSLA",
}


app = FastAPI(
    title="AVA-AI API",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "AVA-AI API is running"
    }


@app.get("/predict/{ticker}")
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