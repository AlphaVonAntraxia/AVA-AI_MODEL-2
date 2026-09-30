from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "model_evaluation.csv"
SKIPPED_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_evaluation_skipped.csv"
)


FEATURE_COLUMNS = [
    "close",
    "MA5",
    "MA20",
    "MA50",
    "price_vs_MA5",
    "price_vs_MA20",
    "price_vs_MA50",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "volume_change",
    "volume_vs_MA20",
    "return_lag1",
    "return_lag2",
    "return_lag3",
    "return_lag5",
    "return_lag10",
    "rolling_return_5",
    "rolling_return_10",
    "rolling_return_20",
    "RSI_14",
    "MACD",
    "MACD_signal",
    "MACD_histogram",
]


WALK_SPLITS = [
    (150, 150, 210),
    (210, 210, 270),
    (270, 270, 330),
    (330, 330, 390),
    (390, 390, 450),
]


RANDOM_STATE = 42


def build_features(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        parse_dates=["datetime"],
    )

    df = (
        df.sort_values("datetime")
        .set_index("datetime")
    )

    df["return"] = df["close"].pct_change()

    df["return_lag1"] = df["return"].shift(1)
    df["return_lag2"] = df["return"].shift(2)
    df["return_lag3"] = df["return"].shift(3)
    df["return_lag5"] = df["return"].shift(5)
    df["return_lag10"] = df["return"].shift(10)

    df["MA5"] = df["close"].rolling(5).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()

    df["price_vs_MA5"] = (
        df["close"] / df["MA5"] - 1
    )

    df["price_vs_MA20"] = (
        df["close"] / df["MA20"] - 1
    )

    df["price_vs_MA50"] = (
        df["close"] / df["MA50"] - 1
    )

    df["rolling_return_5"] = (
        df["return"].rolling(5).mean()
    )

    df["rolling_return_10"] = (
        df["return"].rolling(10).mean()
    )

    df["rolling_return_20"] = (
        df["return"].rolling(20).mean()
    )

    df["volatility_5"] = (
        df["return"].rolling(5).std()
    )

    df["volatility_10"] = (
        df["return"].rolling(10).std()
    )

    df["volatility_20"] = (
        df["return"].rolling(20).std()
    )

    df["volume_change"] = (
        df["volume"].pct_change()
    )

    df["volume_MA20"] = (
        df["volume"].rolling(20).mean()
    )

    df["volume_vs_MA20"] = (
        df["volume"] / df["volume_MA20"] - 1
    )

    delta = df["close"].diff()

    gain = (
        delta.clip(lower=0)
        .rolling(14)
        .mean()
    )

    loss = (
        (-delta.clip(upper=0))
        .rolling(14)
        .mean()
    )

    rs = gain / loss

    df["RSI_14"] = (
        100 - (100 / (1 + rs))
    )

    ema12 = (
        df["close"]
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    df["MACD"] = ema12 - ema26

    df["MACD_signal"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    df["MACD_histogram"] = (
        df["MACD"] - df["MACD_signal"]
    )

    df["target_return"] = (
        df["close"].shift(-1)
        / df["close"]
        - 1
    )

    return df


def evaluate_stock(
    ticker: str,
    file_path: Path,
) -> dict:

    data = build_features(file_path)

    model_data = (
        data[
            FEATURE_COLUMNS
            + ["target_return"]
        ]
        .dropna()
        .copy()
    )

    if len(model_data) < 450:
        raise ValueError(
            f"{ticker}: only "
            f"{len(model_data)} usable rows."
        )

    X = model_data[FEATURE_COLUMNS]
    y = model_data["target_return"]

    model_predictions = []
    actual_returns = []
    naive_predictions = []

    for train_end, test_start, test_end in WALK_SPLITS:

        if test_end > len(model_data):
            continue

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]

        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        # IMPORTANT:
        # Train a brand-new model using ONLY
        # the data available before this test fold.
        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(X_test)

        # Zero-return naive baseline.
        naive = [0.0] * len(y_test)

        model_predictions.extend(
            predictions
        )

        actual_returns.extend(
            y_test.tolist()
        )

        naive_predictions.extend(
            naive
        )

    model_mae = mean_absolute_error(
        actual_returns,
        model_predictions,
    )

    model_rmse = (
        mean_squared_error(
            actual_returns,
            model_predictions,
        )
        ** 0.5
    )

    naive_mae = mean_absolute_error(
        actual_returns,
        naive_predictions,
    )

    naive_rmse = (
        mean_squared_error(
            actual_returns,
            naive_predictions,
        )
        ** 0.5
    )

    return {
        "ticker": ticker,
        "model_MAE": model_mae,
        "naive_MAE": naive_mae,
        "MAE_improvement": (
            naive_mae - model_mae
        ),
        "model_RMSE": model_rmse,
        "naive_RMSE": naive_rmse,
        "RMSE_improvement": (
            naive_rmse - model_rmse
        ),
        "folds": len(WALK_SPLITS),
        "evaluation_rows": len(actual_returns),
    }


def main() -> None:

    results = []
    skipped = []

    data_files = sorted(
        DATA_DIR.glob("*_daily.csv")
    )

    print(
        f"Data files found: "
        f"{len(data_files)}"
    )

    for index, file_path in enumerate(
        data_files,
        start=1,
    ):

        ticker = file_path.stem.replace(
            "_daily",
            "",
        )

        print(
            f"\n[{index}/{len(data_files)}] "
            f"Evaluating {ticker}..."
        )

        try:

            raw_data = pd.read_csv(
                file_path
            )

            if len(raw_data) < 500:
                print(
                    f"Skipping {ticker}: "
                    f"only {len(raw_data)} raw rows."
                )

                skipped.append(
                    {
                        "ticker": ticker,
                        "reason": "LIMITED_HISTORY",
                        "raw_rows": len(raw_data),
                    }
                )

                continue

            metrics = evaluate_stock(
                ticker=ticker,
                file_path=file_path,
            )

            results.append(metrics)

            print(
                f"Model MAE: "
                f"{metrics['model_MAE']:.6f}"
            )

            print(
                f"Naive MAE: "
                f"{metrics['naive_MAE']:.6f}"
            )

            print(
                f"Model RMSE: "
                f"{metrics['model_RMSE']:.6f}"
            )

            print(
                f"Naive RMSE: "
                f"{metrics['naive_RMSE']:.6f}"
            )

        except Exception as exc:

            print(
                f"FAILED {ticker}: {exc}"
            )

            skipped.append(
                {
                    "ticker": ticker,
                    "reason": str(exc),
                }
            )

    results_df = pd.DataFrame(
        results
    )

    skipped_df = pd.DataFrame(
        skipped
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    skipped_df.to_csv(
        SKIPPED_PATH,
        index=False,
    )

    print(
        "\n=== Evaluation Summary ==="
    )

    if not results_df.empty:

        print(
            f"Stocks evaluated: "
            f"{len(results_df)}"
        )

        print(
            f"Average model MAE: "
            f"{results_df['model_MAE'].mean():.6f}"
        )

        print(
            f"Average naive MAE: "
            f"{results_df['naive_MAE'].mean():.6f}"
        )

        print(
            f"Average model RMSE: "
            f"{results_df['model_RMSE'].mean():.6f}"
        )

        print(
            f"Average naive RMSE: "
            f"{results_df['naive_RMSE'].mean():.6f}"
        )

        print(
            f"Stocks beating naive MAE: "
            f"{(results_df['MAE_improvement'] > 0).sum()}"
        )

        print(
            f"Stocks beating naive RMSE: "
            f"{(results_df['RMSE_improvement'] > 0).sum()}"
        )

    print(
        f"Skipped/failed: "
        f"{len(skipped_df)}"
    )

    print(
        f"\nEvaluation saved to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Skipped/failed saved to: "
        f"{SKIPPED_PATH}"
    )


if __name__ == "__main__":
    main()