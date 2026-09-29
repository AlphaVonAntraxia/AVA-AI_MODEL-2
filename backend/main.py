from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.prediction_service import get_prediction
from backend.data.universe.sp500 import search_sp500
from backend.data.universe.coverage import (
    get_data_coverage,
    get_ticker_data_status,
)
from models.availability import get_prediction_availability


class PredictionResponse(BaseModel):
    ticker: str
    predicted_next_day_return: float
    model_version: str
    model_type: str
    prediction_horizon: str


app = FastAPI(
    title="AVA-AI API",
    version="2.0.0",
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

    universe = search_sp500(ticker)

    if universe.empty or ticker not in set(universe["ticker"]):
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' is not in the S&P 500 universe.",
        )

    availability = get_prediction_availability(ticker)

    if not availability["prediction_available"]:
        raise HTTPException(
            status_code=503,
            detail={
                "ticker": ticker,
                "message": "Prediction is not available for this stock yet.",
                "data_available": availability["data_available"],
                "model_available": availability["model_available"],
            },
        )

    return get_prediction(ticker)


@app.get("/stocks/search")
def search_stocks(q: str = ""):
    results = search_sp500(q)

    return {
        "query": q,
        "count": len(results),
        "results": results.to_dict(orient="records"),
    }


@app.get("/stocks/{ticker}/availability")
def stock_availability(ticker: str):
    ticker = ticker.upper()

    universe = search_sp500(ticker)

    if universe.empty or ticker not in set(universe["ticker"]):
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' is not in the S&P 500 universe.",
        )

    return get_prediction_availability(ticker)


@app.get("/stocks/{ticker}/data-status")
def stock_data_status(ticker: str):
    status = get_ticker_data_status(ticker)

    if not status["in_sp500"]:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker.upper()}' is not in the S&P 500 universe.",
        )

    return status


@app.get("/stocks/coverage")
def stock_coverage():
    return get_data_coverage()


@app.get("/stocks")
def list_stocks(q: str = ""):
    results = search_sp500(q)

    stocks = []

    for _, row in results.iterrows():
        ticker = row["ticker"]

        status = get_ticker_data_status(ticker)
        availability = get_prediction_availability(ticker)

        stocks.append(
            {
                "ticker": ticker,
                "company_name": row["company_name"],
                "data_available": status["data_available"],
                "model_available": availability["model_available"],
                "prediction_available": availability["prediction_available"],
            }
        )

    return {
        "query": q,
        "count": len(stocks),
        "results": stocks,
    }