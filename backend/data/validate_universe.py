from pathlib import Path

import pandas as pd

from backend.data.universe.sp500 import load_sp500_universe


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


REQUIRED_COLUMNS = {
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def validate_ticker(ticker: str) -> dict:
    ticker = ticker.upper()
    path = RAW_DATA_DIR / f"{ticker}_daily.csv"

    result = {
        "ticker": ticker,
        "file_exists": path.exists(),
        "rows": 0,
        "missing_values": 0,
        "duplicate_dates": 0,
        "invalid_numeric_values": 0,
        "invalid_ohlc": 0,
        "date_min": None,
        "date_max": None,
        "status": "FAIL",
        "error": None,
    }

    if not path.exists():
        result["error"] = "File missing"
        return result

    try:
        df = pd.read_csv(path)

        missing_columns = REQUIRED_COLUMNS - set(df.columns)

        if missing_columns:
            result["error"] = (
                "Missing columns: "
                + ", ".join(sorted(missing_columns))
            )
            return result

        result["rows"] = len(df)

        result["missing_values"] = int(
            df[list(REQUIRED_COLUMNS)].isna().sum().sum()
        )

        dates = pd.to_datetime(
            df["datetime"],
            errors="coerce",
        )

        result["invalid_numeric_values"] = int(
            df[
                ["open", "high", "low", "close", "volume"]
            ]
            .apply(pd.to_numeric, errors="coerce")
            .isna()
            .sum()
            .sum()
        )

        result["duplicate_dates"] = int(
            dates.duplicated().sum()
        )

        if dates.notna().any():
            result["date_min"] = dates.min().date().isoformat()
            result["date_max"] = dates.max().date().isoformat()

        numeric = df[
            ["open", "high", "low", "close"]
        ].apply(pd.to_numeric, errors="coerce")

        result["invalid_ohlc"] = int(
            (
                (numeric["high"] < numeric["open"])
                | (numeric["high"] < numeric["close"])
                | (numeric["high"] < numeric["low"])
                | (numeric["low"] > numeric["open"])
                | (numeric["low"] > numeric["close"])
            ).sum()
        )

        # Classify datasets based on data quality and history length.
        if (
            result["missing_values"] == 0
            and result["duplicate_dates"] == 0
            and result["invalid_numeric_values"] == 0
            and result["invalid_ohlc"] == 0
            and dates.notna().all()
        ):
            if result["rows"] >= 500:
                result["status"] = "PASS"
            else:
                result["status"] = "LIMITED_HISTORY"
        else:
            result["status"] = "FAIL"

    except Exception as exc:
        result["error"] = str(exc)

    return result


def validate_universe() -> pd.DataFrame:
    universe = load_sp500_universe()

    results = []

    for ticker in universe["ticker"]:
        ticker = ticker.upper()

        print(f"Validating {ticker}...")

        results.append(
            validate_ticker(ticker)
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    results = validate_universe()

    print("\n" + "=" * 60)
    print("AVA-AI V2 DATA VALIDATION")
    print("=" * 60)

    print(f"Stocks checked: {len(results)}")

    print(
        f"PASS: {(results['status'] == 'PASS').sum()}"
    )

    print(
        f"LIMITED_HISTORY: "
        f"{(results['status'] == 'LIMITED_HISTORY').sum()}"
    )

    print(
        f"FAIL: {(results['status'] == 'FAIL').sum()}"
    )

    print("\nRows:")

    print(
        f"Minimum: {results['rows'].min()}"
    )

    print(
        f"Maximum: {results['rows'].max()}"
    )

    print(
        f"Average: {results['rows'].mean():.1f}"
    )

    print("\nDate ranges:")

    valid_dates = results[
        results["date_min"].notna()
        & results["date_max"].notna()
    ]

    if not valid_dates.empty:
        print(
            f"Earliest: {valid_dates['date_min'].min()}"
        )

        print(
            f"Latest: {valid_dates['date_max'].max()}"
        )

    problems = results[
        results["status"] != "PASS"
    ]

    if not problems.empty:
        print("\nStocks requiring attention:")

        print(
            problems[
                [
                    "ticker",
                    "status",
                    "rows",
                    "missing_values",
                    "duplicate_dates",
                    "invalid_numeric_values",
                    "invalid_ohlc",
                    "error",
                ]
            ].to_string(index=False)
        )
    else:
        print(
            "\nAll 503 datasets passed validation."
        )
