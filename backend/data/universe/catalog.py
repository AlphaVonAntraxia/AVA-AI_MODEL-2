from pathlib import Path

import pandas as pd

from backend.data.universe.sp500 import load_sp500_universe


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"


def build_asset_catalog() -> pd.DataFrame:
    universe = load_sp500_universe().copy()

    universe["asset_type"] = "stock"
    universe["data_available"] = universe["ticker"].apply(
        lambda ticker: (
            RAW_DATA_DIR / f"{ticker.upper()}_daily.csv"
        ).exists()
    )

    universe["model_available"] = universe["ticker"].apply(
        lambda ticker: (
            MODEL_DIR / f"{ticker.upper()}_random_forest.joblib"
        ).exists()
    )

    universe["prediction_available"] = (
        universe["data_available"]
        & universe["model_available"]
    )

    return universe


if __name__ == "__main__":
    catalog = build_asset_catalog()

    catalog.to_csv(
    PROJECT_ROOT / "data" / "asset_catalog.csv",
    index=False,
    )

    print("\n" + "=" * 60)
    print("AVA-AI V2 ASSET CATALOG")
    print("=" * 60)

    print(f"Assets: {len(catalog)}")
    print(
        f"Data available: "
        f"{catalog['data_available'].sum()}"
    )
    print(
        f"Models available: "
        f"{catalog['model_available'].sum()}"
    )
    print(
        f"Predictions available: "
        f"{catalog['prediction_available'].sum()}"
    )

    print("\nSample:")
    print(catalog.head().to_string(index=False))