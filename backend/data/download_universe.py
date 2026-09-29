from pathlib import Path
import time

from backend.data.market_data import get_historical_data
from backend.data.universe.sp500 import load_sp500_universe


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_ticker(
    symbol: str,
    outputsize: int = 500,
    retries: int = 3,
    retry_delay: int = 5,
) -> Path:
    symbol = symbol.upper()

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            print(
                f"Downloading {symbol} "
                f"(attempt {attempt}/{retries})..."
            )

            df = get_historical_data(
                symbol=symbol,
                interval="1day",
                outputsize=outputsize,
            )

            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

            output_path = RAW_DATA_DIR / f"{symbol}_daily.csv"
            df.to_csv(output_path)

            print(
                f"Saved {symbol}: {len(df)} rows -> {output_path}"
            )

            return output_path

        except Exception as exc:
            last_error = exc

            print(
                f"Failed {symbol} on attempt "
                f"{attempt}/{retries}: {exc}"
            )

            if attempt < retries:
                print(
                    f"Waiting {retry_delay} seconds before retry..."
                )
                time.sleep(retry_delay)

    raise RuntimeError(
        f"Failed to download {symbol} after {retries} attempts: "
        f"{last_error}"
    )


def download_universe(
    symbols=None,
    outputsize: int = 500,
    limit: int | None = None,
    skip_existing: bool = True,
    request_delay: int = 2,
) -> None:
    if symbols is None:
        universe = load_sp500_universe()
        symbols = universe["ticker"].tolist()

    if limit is not None:
        symbols = symbols[:limit]

    print(f"Stocks selected for download: {len(symbols)}")

    for index, symbol in enumerate(symbols, start=1):
        symbol = symbol.upper()

        print(f"\n[{index}/{len(symbols)}] {symbol}")

        output_path = RAW_DATA_DIR / f"{symbol}_daily.csv"

        if skip_existing and output_path.exists():
            print(f"Skipping {symbol}: data already exists")
            continue

        try:
            download_ticker(
                symbol,
                outputsize=outputsize,
            )
        except Exception as exc:
            print(f"Giving up on {symbol}: {exc}")

        if index < len(symbols):
            print(
                f"Waiting {request_delay} seconds before next stock..."
            )
            time.sleep(request_delay)


if __name__ == "__main__":
    download_universe(limit=5)
