# AVA-AI V2 Base Model

## 1. Overview

AVA-AI V2 is the project's first broad, production-oriented forecasting foundation.

V2 expands the original six-stock research prototype into an S&P 500-scale system with:

- 503 S&P 500 assets
- Automated historical-data acquisition
- Data-quality validation
- Asset cataloging
- Model training and registration
- Leakage-free walk-forward evaluation
- Naive baseline comparison
- Prediction availability tracking
- Data-freshness metadata
- FastAPI backend integration
- Frontend prediction dashboard

The system predicts the next trading day's percentage return.

AVA-AI V2 is a research and engineering system. Its current evaluation does not demonstrate broad predictive superiority over a naive baseline and should not be interpreted as a validated trading strategy or investment recommendation.

---

## 2. V2 Asset Universe

The V2 universe contains the 503 constituents represented by the project's S&P 500 universe file.

The universe is stored in:

`backend/data/universe/sp500.csv`

The asset catalog is generated at:

`data/asset_catalog.csv`

Each catalog entry tracks:

- ticker
- company name
- asset type
- data availability
- model availability
- prediction availability

### Current Coverage

| Metric | Count |
|---|---:|
| S&P 500 assets | 503 |
| Historical datasets available | 503 |
| Trained models | 499 |
| Prediction-ready assets | 499 |
| Assets without models | 4 |

Four assets have insufficient historical data for the current five-fold training procedure and are therefore intentionally excluded from model training.

These are currently recorded as:

- FDXF
- HONA
- Q
- SNDK

Their raw data passed the available data-quality checks, but the historical length is insufficient for the current training architecture.

---

## 3. Historical Market Data

V2 uses daily historical market data containing:

- datetime
- open
- high
- low
- close
- volume

Raw datasets are stored under:

`data/raw/`

Raw market-data files are intentionally excluded from Git because of their size.

The project uses Twelve Data for historical data acquisition.

The market-data client is implemented in:

`backend/data/market_data.py`

The universe downloader is implemented in:

`backend/data/download_universe.py`

The downloader supports:

- configurable output size
- retries
- retry delays
- request delays
- skipping existing datasets
- complete S&P 500 universe downloads

The final S&P 500 acquisition achieved 100% dataset availability.

---

## 4. Data Validation

V2 includes automated validation before model training.

Validation checks include:

- required columns
- missing values
- duplicate dates
- invalid numeric values
- invalid OHLC relationships
- historical row count
- earliest available date
- latest available date

The validation process found:

| Status | Assets |
|---|---:|
| PASS | 499 |
| LIMITED_HISTORY / CHECK | 4 |
| FAIL | 0 |

The four limited-history datasets passed the underlying data-quality checks but do not contain enough observations for the current model-training procedure.

This distinction is intentional: insufficient history is treated differently from corrupted market data.

---

## 5. Feature Engineering

The current V2 production feature set contains 24 technical features.

```text
close
MA5
MA20
MA50
price_vs_MA5
price_vs_MA20
price_vs_MA50
volatility_5
volatility_10
volatility_20
volume_change
volume_vs_MA20
return_lag1
return_lag2
return_lag3
return_lag5
return_lag10
rolling_return_5
rolling_return_10
rolling_return_20
RSI_14
MACD
MACD_signal
MACD_histogram