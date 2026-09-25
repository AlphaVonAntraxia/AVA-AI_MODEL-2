from pathlib import Path

import pandas as pd

from data.market_data import get_historical_data


def validate_data(df: pd.DataFrame) -> None:
    """Run basic validation checks on market data."""

    print("\n--- Data Validation ---")

    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Check for missing values
    missing_values = df.isnull().sum()

    print("\nMissing values:")
    print(missing_values)

    # Check for duplicate timestamps
    duplicate_rows = df.index.duplicated().sum()

    print(f"\nDuplicate timestamps: {duplicate_rows}")

    # Check chronological ordering
    is_sorted = df.index.is_monotonic_increasing

    print(f"Chronologically sorted: {is_sorted}")

    # Check for invalid OHLC relationships
    invalid_high = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
    invalid_low = (df["low"] > df[["open", "close"]].min(axis=1)).sum()

    print(f"Invalid high values: {invalid_high}")
    print(f"Invalid low values: {invalid_low}")

    # Check for non-positive prices
    non_positive_prices = (
        df[["open", "high", "low", "close"]] <= 0
    ).any(axis=1).sum()

    print(f"Non-positive price rows: {non_positive_prices}")


def main():
    symbol = "AAPL"

    df = get_historical_data(
        symbol=symbol,
        interval="1day",
        outputsize=500,
    )

    print("\n--- Historical Data ---")
    print(df.head())

    print("\n--- Latest Observation ---")
    print(df.iloc[-1])

    validate_data(df)

    # Save the raw dataset locally.
    output_path = Path("data/raw") / f"{symbol}_daily.csv"

    df.to_csv(output_path)

    print(f"\nDataset saved to: {output_path}")


if __name__ == "__main__":
    main()