import os

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("TWELVE_DATA_API_KEY")
BASE_URL = "https://api.twelvedata.com"


def get_historical_data(
    symbol: str,
    interval: str = "1day",
    outputsize: int = 100,
) -> pd.DataFrame:
    """
    Retrieve historical OHLCV data for a stock.

    Parameters
    ----------
    symbol:
        Stock ticker, for example AAPL.

    interval:
        Time interval supported by Twelve Data.
        Default: 1day.

    outputsize:
        Number of historical observations requested.

    Returns
    -------
    pandas.DataFrame
        Historical OHLCV data indexed by datetime.
    """

    if not API_KEY:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY was not found. "
            "Check that your .env file exists and contains the API key."
        )

    endpoint = f"{BASE_URL}/time_series"

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": API_KEY,
    }

    response = requests.get(
        endpoint,
        params=params,
        timeout=30,
    )

    if response.status_code == 429:
        raise RuntimeError(
            "Twelve Data rate limit reached (HTTP 429)."
        )

    response.raise_for_status()

    data = response.json()

    if data.get("status") == "error":
        message = data.get("message", "Unknown API error")
        raise RuntimeError(f"Twelve Data API error: {message}")

    if "values" not in data:
        raise RuntimeError(
            "API response did not contain historical data."
        )

    df = pd.DataFrame(data["values"])

    if df.empty:
        raise RuntimeError(
            f"No historical data returned for {symbol}."
        )

    df["datetime"] = pd.to_datetime(df["datetime"])

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df = df.sort_values("datetime")
    df = df.set_index("datetime")

    return df