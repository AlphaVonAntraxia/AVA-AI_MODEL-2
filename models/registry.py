from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"

CATALOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "asset_catalog.csv"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_training_metrics.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_registry.csv"
)


def build_model_registry() -> pd.DataFrame:
    catalog = pd.read_csv(CATALOG_PATH)

    metrics = pd.read_csv(METRICS_PATH)

    registry = catalog[
        [
            "ticker",
            "company_name",
            "asset_type",
        ]
    ].copy()

    registry["model_version"] = "v2"
    registry["model_type"] = "random_forest"

    registry["model_file"] = registry[
        "ticker"
    ].apply(
        lambda ticker: (
            f"{ticker}_random_forest.joblib"
        )
    )

    registry["data_available"] = registry[
        "ticker"
    ].apply(
        lambda ticker: (
            DATA_DIR
            / f"{ticker}_daily.csv"
        ).exists()
    )

    registry["model_available"] = registry[
        "ticker"
    ].apply(
        lambda ticker: (
            MODEL_DIR
            / f"{ticker}_random_forest.joblib"
        ).exists()
    )

    registry["prediction_available"] = (
        registry["data_available"]
        & registry["model_available"]
    )

    registry = registry.merge(
        metrics[
            [
                "ticker",
                "MAE",
                "RMSE",
                "folds",
                "training_rows",
            ]
        ],
        on="ticker",
        how="left",
    )

    registry["registry_status"] = "READY"

    registry.loc[
        ~registry["data_available"],
        "registry_status",
    ] = "NO_DATA"

    registry.loc[
        registry["data_available"]
        & ~registry["model_available"],
        "registry_status",
    ] = "NO_MODEL"

    return registry


def main() -> None:
    registry = build_model_registry()

    registry.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\n" + "=" * 60
    )
    print(
        "AVA-AI V2 MODEL REGISTRY"
    )
    print(
        "=" * 60
    )

    print(
        f"Assets: "
        f"{len(registry)}"
    )

    print(
        f"Models available: "
        f"{registry['model_available'].sum()}"
    )

    print(
        f"Prediction available: "
        f"{registry['prediction_available'].sum()}"
    )

    print(
        "\nRegistry status:"
    )

    print(
        registry[
            "registry_status"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nSample:"
    )

    print(
        registry.head()
        .to_string(index=False)
    )

    print(
        f"\nRegistry saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()