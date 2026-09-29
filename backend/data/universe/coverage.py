from pathlib import Path

from backend.data.universe.sp500 import load_sp500_universe


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def get_data_coverage() -> dict:
    universe = load_sp500_universe()

    available = {
        path.stem.replace("_daily", "")
        for path in RAW_DATA_DIR.glob("*_daily.csv")
    }

    universe_tickers = set(universe["ticker"])

    available_in_universe = universe_tickers & available
    missing = universe_tickers - available

    return {
        "universe_size": len(universe_tickers),
        "data_available": len(available_in_universe),
        "data_missing": len(missing),
        "coverage_percent": (
            len(available_in_universe)
            / len(universe_tickers)
            * 100
        ),
        "available_tickers": sorted(available_in_universe),
        "missing_tickers": sorted(missing),
    }


def get_ticker_data_status(ticker: str) -> dict:
    ticker = ticker.upper()

    universe = load_sp500_universe()
    universe_tickers = set(universe["ticker"])

    if ticker not in universe_tickers:
        return {
            "ticker": ticker,
            "in_sp500": False,
            "data_available": False,
        }

    data_path = RAW_DATA_DIR / f"{ticker}_daily.csv"

    return {
        "ticker": ticker,
        "in_sp500": True,
        "data_available": data_path.exists(),
    }


if __name__ == "__main__":
    coverage = get_data_coverage()

    print(f"Universe size: {coverage['universe_size']}")
    print(f"Data available: {coverage['data_available']}")
    print(f"Data missing: {coverage['data_missing']}")
    print(f"Coverage: {coverage['coverage_percent']:.2f}%")

    print("\nTicker status examples:")

    for ticker in ["AAPL", "NVDA", "KO", "NOTREAL"]:
        print(get_ticker_data_status(ticker))