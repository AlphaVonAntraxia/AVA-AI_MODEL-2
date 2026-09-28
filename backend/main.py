from fastapi import FastAPI

from backend.prediction_service import get_prediction


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
    return get_prediction(ticker)