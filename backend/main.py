from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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



app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://ava-ai.local:5500",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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