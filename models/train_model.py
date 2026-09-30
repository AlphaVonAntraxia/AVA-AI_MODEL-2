from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "data"


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
MIN_RAW_ROWS = 500


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

    # Predict the next trading day's return.
    df["target_return"] = (
        df["close"].shift(-1)
        / df["close"]
        - 1
    )

    return df


def train_stock_model(
    ticker: str,
    data: pd.DataFrame,
) -> tuple[RandomForestRegressor, dict]:

    model_data = (
        data[FEATURE_COLUMNS + ["target_return"]]
        .dropna()
        .copy()
    )

    if len(model_data) < 450:
        raise ValueError(
            f"{ticker}: only {len(model_data)} usable rows "
            "available; 450 required for V2 walk-forward validation."
        )

    X = model_data[FEATURE_COLUMNS]
    y = model_data["target_return"]

    fold_mae = []
    fold_rmse = []

    for train_end, test_start, test_end in WALK_SPLITS:

        if test_end > len(model_data):
            continue

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]

        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

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

        fold_mae.append(
            mean_absolute_error(
                y_test,
                predictions,
            )
        )

        fold_rmse.append(
            mean_squared_error(
                y_test,
                predictions,
            ) ** 0.5
        )

    if not fold_mae:
        raise ValueError(
            f"{ticker}: no valid walk-forward folds."
        )

    final_model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    final_model.fit(
        X,
        y,
    )

    metrics = {
        "ticker": ticker,
        "MAE": sum(fold_mae) / len(fold_mae),
        "RMSE": sum(fold_rmse) / len(fold_rmse),
        "folds": len(fold_mae),
        "training_rows": len(model_data),
    }

    return final_model, metrics


def main() -> None:
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []
    skipped = []

    files = sorted(
        DATA_DIR.glob("*_daily.csv")
    )

    print(
        f"Stocks found: {len(files)}"
    )

    for index, file_path in enumerate(
        files,
        start=1,
    ):
        ticker = file_path.stem.replace(
            "_daily",
            "",
        )

        print(
            f"\n[{index}/{len(files)}] Training {ticker}..."
        )

        raw_data = pd.read_csv(
            file_path
        )

        raw_rows = len(raw_data)

        if raw_rows < MIN_RAW_ROWS:
            print(
                f"Skipping {ticker}: "
                f"insufficient history "
                f"({raw_rows} raw rows; "
                f"{MIN_RAW_ROWS} required)"
            )

            skipped.append(
                {
                    "ticker": ticker,
                    "reason": "LIMITED_HISTORY",
                    "raw_rows": raw_rows,
                }
            )

            continue

        try:
            data = build_features(
                file_path
            )

            model, metrics = train_stock_model(
                ticker=ticker,
                data=data,
            )

            model_path = (
                MODEL_DIR
                / f"{ticker}_random_forest.joblib"
            )

            joblib.dump(
                model,
                model_path,
            )

            results.append(metrics)

            print(
                f"Model saved: {model_path}"
            )

            print(
                f"MAE: {metrics['MAE']:.6f}"
            )

            print(
                f"RMSE: {metrics['RMSE']:.6f}"
            )

            print(
                f"Folds: {metrics['folds']}"
            )

            print(
                f"Training rows: "
                f"{metrics['training_rows']}"
            )

        except Exception as exc:
            print(
                f"FAILED {ticker}: {exc}"
            )

            skipped.append(
                {
                    "ticker": ticker,
                    "reason": str(exc),
                    "raw_rows": raw_rows,
                }
            )

    results_df = pd.DataFrame(
        results
    )

    skipped_df = pd.DataFrame(
        skipped
    )

    metrics_path = (
        METRICS_DIR
        / "model_training_metrics.csv"
    )

    skipped_path = (
        METRICS_DIR
        / "model_training_skipped.csv"
    )

    results_df.to_csv(
        metrics_path,
        index=False,
    )

    skipped_df.to_csv(
        skipped_path,
        index=False,
    )

    print(
        "\n=== Training Summary ==="
    )

    if not results_df.empty:
        print(
            results_df.to_string(
                index=False
            )
        )

        print(
            f"\nSuccessfully trained: "
            f"{len(results_df)}"
        )

        print(
            f"Average MAE: "
            f"{results_df['MAE'].mean():.6f}"
        )

        print(
            f"Average RMSE: "
            f"{results_df['RMSE'].mean():.6f}"
        )

    print(
        f"\nSkipped/failed: "
        f"{len(skipped_df)}"
    )

    if not skipped_df.empty:
        print(
            skipped_df.to_string(
                index=False
            )
        )

    print(
        f"\nTraining metrics saved to: "
        f"{metrics_path}"
    )

    print(
        f"Skipped/failed stocks saved to: "
        f"{skipped_path}"
    )


if __name__ == "__main__":
    main()